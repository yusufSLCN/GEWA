from tpp_dataset import TPPDataset
from visualize_tpp_dataset import show_grasp_and_edge_predictions
import argparse
from torch_geometric.loader import DataListLoader, DataLoader
import numpy as np
import time


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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    parser.add_argument('-i', '--sample_idx', type=int, default=0)
    args = parser.parse_args()
    model_name = args.model_name

    # Initialize a run dont upload the run info

    # run = wandb.init(project="Grasp", job_type="download_model", notes="inference")

    # # idx 4, 5, 6, 7, 13, 18
    # # Access and download model. Returns path to downloaded artifact
    # # downloaded_model_path = run.use_model(name="TppNet_nm_500__bs_32.pth_epoch_130_acc_0.962_recall_0.509_prec_0.270.pth:v0")
    # #contrastive loss
    # # downloaded_model_path = run.use_model(name="TppNet_nm_2000__bs_32.pth_epoch_450_acc_0.972_recall_0.517_prec_0.400.pth:v0") 
    # # downloaded_model_path = run.use_model(name="TppNet_nm_2000__bs_32.pth_epoch_390_acc_0.970_recall_0.574_prec_0.381.pth:v0")
    # # wo contrastive loss
    # # downloaded_model_path = run.use_model(name="TppNet_nm_2000__bs_32.pth_epoch_400_acc_0.972_recall_0.523_prec_0.413.pth:v0")
    # #global embeddings
    # # downloaded_model_path = run.use_model(name="TppNet_nm_2000__bs_32.pth_epoch_420_acc_0.972_recall_0.578_prec_0.412.pth:v0")
    # # downloaded_model_path = run.use_model(name="TppNet_nm_2000__bs_32.pth_epoch_490_acc_0.972_recall_0.576_prec_0.420.pth:v0")
    
    # # with grasp head
    # downloaded_model_path = run.use_model(name="TppNet_nm_1000__bs_8.pth_epoch_90_acc_0.93_recall_0.62.pth:v0")
    
    # print(downloaded_model_path)

    # model_path = downloaded_model_path

    # # load the GraspNet model and run inference then display the gripper pose
    # model = TppNet()
    # model = nn.DataParallel(model)
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    from create_tpp_dataset import save_split_samples
    train_paths, val_paths = save_split_samples('../data', 100, dataset_name="tpp_effdict")

    dataset = TPPDataset(val_paths, return_pair_dict=True)
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
    # grasp_pred, selected_edge_idxs, mid_edge_pos, grasp_target, num_valid_grasps, pair_classification_pred, pair_dot_product = model(data)
    # pair_classification_pred, pair_dot_product, _, _ = model(data)

    #display grasps
    pair_scores = data.pair_scores.numpy()
    pos = data.pos.numpy()  
    triu_indeces = dataset.triu_indices

    selected_edge_idxs = np.where(pair_scores > 0.5)[0]
    np.random.seed(0)
    selected_edge_idxs = np.random.choice(selected_edge_idxs, 10, replace=False)
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

        grasp_pred.append(grasp)


        # key = frozenset((i, j))
        # if key in grasp_dict:
        #     # print(key)
        #     # print(grasp_dict)
        #     grasp = grasp_dict[key][0]
        #     grasp_pred.append(grasp)
            

    grasp_pred = np.array(grasp_pred)
    print(grasp_pred.shape)
    print(selected_edge_idxs.shape)



    show_grasp_and_edge_predictions(pos, grasp_dict, selected_edge_idxs, grasp_pred,
                                    dataset.triu_indices, sample_info, num_grasp_to_show=5)