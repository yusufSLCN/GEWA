import torch
import torch.nn as nn
from torch_geometric.data import Data
# from GraspNet import GraspNet
from deprecated.TppAngleNet import TppAngleNet
from tpp_dataset import TPPDataset
from visualize_tpp_dataset import show_pair_edges, show_grasp_and_edge_predictions
import argparse
import wandb
from torch_geometric.loader import DataListLoader, DataLoader
import numpy as np
from torcheval.metrics.functional.classification import binary_recall
# from sklearn.metrics import recall_score
from metrics import count_correct_approach_scores, check_batch_grasp_success_rate_per_point, check_succces_with_whole_dataset
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    parser.add_argument('-i', '--sample_idx', type=int, default=0)
    args = parser.parse_args()
    model_name = args.model_name

    # Initialize a run dont upload the run info

    run = wandb.init(project="Grasp", job_type="download_model", notes="inference")

    # idx 4, 5, 6, 7, 13, 18

    # downloaded_model_path = run.use_model(name="TppAngleNet_nm_100__bs_8.pth_epoch_900_acc_0.97_recall_0.36.pth:v0")
    downloaded_model_path = run.use_model(name="TppAngleNet_nm_100__bs_8.pth_epoch_490_acc_0.91_recall_0.57.pth:v0")
    #with translation head
    # downloaded_model_path = run.use_model(name="TppAngleNet_nm_100__bs_8.pth_epoch_980_acc_0.97_recall_0.35.pth:v0")
    
    
    print(downloaded_model_path)

    model_path = downloaded_model_path

    # load the GraspNet model and run inference then display the gripper pose
    model = TppAngleNet()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    from create_tpp_dataset import save_split_samples
    train_paths, val_paths = save_split_samples('../data', 400, dataset_name="tpp_effdict")

    dataset = TPPDataset(val_paths, return_pair_dict=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    samlpe_idx = args.sample_idx
    # data = data_loader[samlpe_idx]

    # print(data)
    model.device = 'cpu'
    model.eval()

    for i, d in enumerate(data_loader):
        t = time.time()
        data = d
        # print(data.sample_info)
        print(f"Time to load data: {time.time() - t}")
        if i == samlpe_idx:
            break
    print(data.sample_info)
    grasp_pred, selected_edge_idxs, mid_edge_pos, grasp_target, num_valid_grasps, pair_classification_pred, pair_dot_product = model(data)
    # pair_classification_pred, pair_dot_product, _, _ = model(data)

    test_pair_accuracy = count_correct_approach_scores(pair_classification_pred, data.pair_scores)
    test_pair_accuracy = test_pair_accuracy / len(data.pair_scores)
    print(f"Test pair accuracy: {test_pair_accuracy}")

    pair_classification_pred = torch.flatten(pair_classification_pred)
    binary_pair_scores_gt = torch.flatten(data.pair_scores).int()
    test_recall = binary_recall(pair_classification_pred, binary_pair_scores_gt)
    print(f"Test pair recall: {test_recall}")

    # grasp_pred = grasp_pred.detach().numpy()
    # grasp_pred = grasp_pred.reshape(1, -1, 1, 4, 4)
    # grasp_target = grasp_target.detach().numpy()
    # grasp_target = grasp_target.reshape(1, -1, 10, 4, 4)
    # # print(grasp_pred.shape, grasp_target.shape)
    # num_valid_grasps = num_valid_grasps.detach().numpy()
    # grasp_success = check_batch_grasp_success_rate_per_point(grasp_pred, grasp_target, 0.03,
                                                                # np.deg2rad(30), num_valid_grasps)
    # grasp_success = grasp_success / len(grasp_pred)
    grasp_dict = data.y[0][0]
    grasp_pred = grasp_pred.detach().numpy()
    grasp_pred = grasp_pred.reshape( -1, 1, 4, 4)
    # test_grasp_pred = grasp_dict[list(grasp_dict.keys())[0]][0].reshape(1, 1, 4, 4)
    # print(f"{test_grasp_pred.shape}")
    grasp_success = check_succces_with_whole_dataset(grasp_pred, grasp_dict, 0.03, np.deg2rad(30))
    print(f"Grasp success rate: {grasp_success}")
    # Display the result
    pair_scores = data.pair_scores.numpy()
    pos = data.pos.numpy()  
    pred_pair_scores = pair_classification_pred.squeeze().detach().numpy()

    print(f"min: {np.min(pred_pair_scores)}, max: {np.max(pred_pair_scores)}")
    threshold = 0.5
    num_pred_pairs = np.sum(pred_pair_scores > threshold)
    print(f"Number of predicted pairs: {num_pred_pairs}/{len(pred_pair_scores)}")

    num_gt_pairs = np.sum(pair_scores > 0)
    print(f"Number of ground truth pairs: {num_gt_pairs}/{len(pair_scores)}")

    show_pair_edges(pos, pred_pair_scores, dataset.triu_indices, threshold=threshold)
    show_pair_edges(pos, pair_scores, dataset.triu_indices)

    #display grasps
    print(selected_edge_idxs.shape)
    grasp_pred = grasp_pred.reshape(selected_edge_idxs.shape[1], 16)

    sample_info = data.sample_info
    selected_edge_idxs = selected_edge_idxs.squeeze()
    # print(selected_edge_idxs)
    # print(np.where(pred_pair_scores > threshold))

    show_grasp_and_edge_predictions(pos, grasp_dict, selected_edge_idxs, grasp_pred,
                                    dataset.triu_indices, sample_info, num_grasp_to_show=5)