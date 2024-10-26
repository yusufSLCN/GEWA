from tpp_dataset import TPPDataset
from visualize_tpp_dataset import show_grasp_and_edge_predictions
import argparse
from torch_geometric.loader import DataListLoader, DataLoader
import numpy as np
import time
from metrics import check_batch_success_with_whole_gewa_dataset
from TppAngleNet import TppAngleNet

import numpy as np

def rotate_vector(v, k, theta):
    """
    Rotate vector v around unit vector k by angle theta (in radians).
    """
    # Ensure k is a unit vector
    k = k / np.linalg.norm(k)
    
    # Compute the rotated vector using Rodrigues' formula
    v_rot = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + k * np.dot(k, v) * (1 - np.cos(theta))
    
    return v_rot





def visualize_angle_sampling(data, dataset):
    #display grasps
    pair_scores = data.pair_scores.numpy()
    pos = data.pos.numpy()  
    triu_indeces = dataset.triu_indices
    num_grasp_samples = 200
    selected_edge_idxs = np.where(pair_scores > 0.5)[0]
    np.random.seed(0)
    selected_edge_idxs = np.random.choice(selected_edge_idxs, num_grasp_samples, replace=False)
    grasp_dict = data.y[0][0]
    sample_info = data.sample_info
    grasp_pred = []
    for edge_idx in selected_edge_idxs:
        i = triu_indeces[0][edge_idx]
        j = triu_indeces[1][edge_idx]
        node_i = pos[i]
        node_j = pos[j]
        grasp_axis = node_j - node_i
        grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)
        approach_axis = np.array([0, 0, 1])
        approach_axis = np.cross(grasp_axis, approach_axis)
        approach_axis = approach_axis / np.linalg.norm(approach_axis)

        angle = 0
        # angle = -np.pi / 2
        approach_axis = rotate_vector(approach_axis , grasp_axis, angle)
        approach_axis = approach_axis / np.linalg.norm(approach_axis)


        normal_axis = np.cross(grasp_axis, approach_axis)
        normal_axis = normal_axis / np.linalg.norm(normal_axis)


        mid_point = (node_i + node_j) / 2

        translation = mid_point - approach_axis * 1.12169998e-01
        grasp = np.eye(4)
        grasp[:3, 3] = translation
        grasp[:3, 0] = grasp_axis
        grasp[:3, 1] = normal_axis
        grasp[:3, 2] = approach_axis
        # gripper_height = np.array([0, 0, 1.12169998e-01, 1])

        # grasp_pred.append(grasp)


        key = frozenset((i, j))
        if key in grasp_dict:
            # print(key)
            # print(grasp_dict)
            grasp = grasp_dict[key][0]
            grasp[:3, 3] -= np.array(data.sample_info["mean"]).reshape(3)
            grasp_pred.append(grasp)
            

    grasp_pred = np.array(grasp_pred)
    print(grasp_pred.shape)
    print(selected_edge_idxs.shape)


    mean = np.array(data.sample_info["mean"]).reshape(3)
    show_grasp_and_edge_predictions(pos, grasp_dict, selected_edge_idxs, grasp_pred,
                                    dataset.triu_indices, sample_info, num_grasp_to_show=5, mean=mean)
    grasp_gt_paths = data.sample_info['grasps']
    mean = mean.reshape(1, 3)
    grasp_pred = grasp_pred.reshape(-1, num_grasp_samples, 1, 4, 4)
    print(grasp_pred.shape, mean.shape, grasp_gt_paths)
    success_rate = check_batch_success_with_whole_gewa_dataset(grasp_pred, 0.03, np.deg2rad(30), grasp_gt_paths, mean)
    print("Success rate: ", success_rate)


def test_TppAngleNet(data):
    import torch
    model = TppAngleNet(num_grasp_sample=200,
                max_num_grasps=10, only_classifier=False, normalize=True)
    grasp_pred, selected_edge_idxs, mid_edge_pos, grasp_axises, grasp_target, num_valid_grasps, pair_classification_pred, mlp_out_ij = model(data)
    pair_scores_gt = data.pair_scores
    pair_scores_gt = pair_scores_gt.reshape(-1, model.num_pairs)
    
    print(grasp_pred.shape, grasp_axises.shape)
    pred_grasp_axis = grasp_pred[:, :3, 0]
    print(pred_grasp_axis[0], grasp_axises[0])

    # pred_grasp_axis = pred_grasp_axis.reshape(-1, 3)
    dot_product = torch.sum(pred_grasp_axis * grasp_axises, dim=-1)
    squared_dot = dot_product ** 2
    grasp_axis_loss = -torch.mean(squared_dot)
    print("Axis loss: ", grasp_axis_loss)

    pos = data.pos.numpy()
    grasp_dict = data.y[0][0]
    sample_info = data.sample_info
    mean = np.array(data.sample_info["mean"]).reshape(3)
    selected_edge_idxs = selected_edge_idxs.reshape(-1)
    grasp_pred = grasp_pred.detach().numpy()
    show_grasp_and_edge_predictions(pos, grasp_dict, selected_edge_idxs, grasp_pred,
                                    dataset.triu_indices, sample_info, num_grasp_to_show=5, mean=mean)
    grasp_gt_paths = data.sample_info['grasps']
    mean = mean.reshape(1, 3)
    grasp_pred = grasp_pred.reshape(-1, model.num_grasp_sample, 1, 4, 4)
    success_rate = check_batch_success_with_whole_gewa_dataset(grasp_pred, 0.03, np.deg2rad(30), grasp_gt_paths, mean)
    print("Success rate: ", success_rate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--sample_idx', type=int, default=0)
    args = parser.parse_args()

    from create_tpp_dataset import save_contactnet_split_samples
    train_paths, val_paths = save_contactnet_split_samples('../data', 1200, dataset_name="tpp_effdict_nomean_wnormals")

    dataset = TPPDataset(val_paths, return_pair_dict=True, normalize=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    samlpe_idx = args.sample_idx
    # data = data_loader[samlpe_idx]

    # # print(data)
    # model.device = 'cpu'
    # model.eval()

    for i, d in enumerate(data_loader):
        t = time.time()
        data = d
        # print(data.sample_info)
        print(f"Time to load data: {time.time() - t}")
        if i == samlpe_idx:
            break
    print(data.sample_info)

    # visualize_angle_sampling(data, dataset)
    test_TppAngleNet(data)
    # grasp_pred, selected_edge_idxs, mid_edge_pos, grasp_target, num_valid_grasps, pair_classification_pred, pair_dot_product = model(data)
    # pair_classification_pred, pair_dot_product, _, _ = model(data)