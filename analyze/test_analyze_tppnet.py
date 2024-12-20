import torch
import torch.nn as nn
from torch_geometric.data import Data
from models.TppNet import TppNet
from deprecated.TppNetOld import TppNetOld
from dataset.tpp_dataset import TPPDataset
from dataset.create_tpp_dataset import save_contactnet_split_samples

from utils.visualize_tpp_dataset import show_pair_edges, show_grasp_and_edge_predictions
import argparse
import wandb
from torch_geometric.loader import DataListLoader, DataLoader
import numpy as np
from torcheval.metrics.functional.classification import binary_recall
# from sklearn.metrics import recall_score
from utils.metrics import count_correct_approach_scores, check_succces_with_whole_dataset, check_succces_with_whole_gewa_dataset
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    args = parser.parse_args()
    model_name = args.model_name

    # Initialize a run dont upload the run info

    run = wandb.init(project="Grasp", job_type="download_model", notes="inference")

    # idx 4, 5, 6, 7, 13, 18
    #calcualted translation
    # calculated trans
    # contact split
    # downloaded_model_path = run.use_model(name="TppNet_nm_1200__bs_4.pth_epoch_540_success_0.60_acc_0.95_recall_0.61.pth:v0")

    #100 tip loss 
    # downloaded_model_path = run.use_model(name="TppNet_nm_1200__bs_4.pth_epoch_330_success_0.66_acc_0.97_recall_0.34.pth:v0")
    #5600 mesh training
    # downloaded_model_path = run.use_model(name="TppNet_nm_5600__bs_4.pth_epoch_410_success_0.71_acc_0.97_recall_0.32.pth:v0")

    #axis loss
    # downloaded_model_path = run.use_model(name="TppNet_nm_1200__bs_4.pth_epoch_570_success_0.56_acc_0.94_recall_0.65.pth:v0")

    #TPPNET NEW, axis loss
    downloaded_model_path = run.use_model(name="TppNet_nm_5600__bs_4.pth_epoch_490_success_0.79_acc_0.94_recall_0.72.pth:v0")

    print(downloaded_model_path)

    model_path = downloaded_model_path

    # load the GraspNet model and run inference then display the gripper pose
    model = TppNet(grasp_dim=7, num_grasp_sample=50, sort_by_score=True, normalize=True, topk=50)
    # model = TppNetOld(grasp_dim= 9, num_grasp_sample=100, sort_by_score=True, normalize=True)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # train_paths, val_paths = save_split_samples('../data', 400, dataset_name="tpp_effdict", contactnet_split=True)
    # train_paths, val_paths = save_split_samples('../data', 400, dataset_name="tpp_effdict_nomean_wnormals", contactnet_split=True)
    train_paths, val_paths = save_contactnet_split_samples('../data', num_mesh=1200, dataset_name="tpp_effdict_nomean_wnormals")
    
    # val_paths = [val_paths[args.sample_idx]]
    dataset = TPPDataset(val_paths, return_pair_dict=True, normalize=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # print(data)
    model.device = 'cpu'
    model.eval()
    success_rates = []
    for i, data in enumerate(data_loader):
        print(f"Processing sample {i}/{len(data_loader)}")
        grasp_pred, selected_edge_idxs, mid_edge_pos, _, grasp_target, num_valid_grasps, pair_classification_pred, pair_dot_product = model(data)
        # pair_classification_pred, pair_dot_product, _, _ = model(data)

        # test_pair_accuracy = count_correct_approach_scores(pair_classification_pred, data.pair_scores)
        # test_pair_accuracy = test_pair_accuracy / len(data.pair_scores)
        # print(f"Test pair accuracy: {test_pair_accuracy}")

        # pair_classification_pred = torch.flatten(pair_classification_pred)
        # binary_pair_scores_gt = torch.flatten(data.pair_scores).int()
        # test_recall = binary_recall(pair_classification_pred, binary_pair_scores_gt)
        # print(f"Test pair recall: {test_recall}")


        grasp_dict = data.y[0][0]
        grasp_pred = grasp_pred.detach().numpy()
        grasp_pred = grasp_pred.reshape( -1, 1, 4, 4)
        # print(f"Grasp pred shape: {grasp_pred.shape}")
        # test_grasp_pred = grasp_dict[list(grasp_dict.keys())[0]][0].reshape(1, 1, 4, 4)
        # print(f"{test_grasp_pred.shape}")

        grasp_gt_path = data.sample_info['grasps'][0]
        # pointcloud_mean = pointcloud_mean.detach().cpu().numpy()
        pointcloud_mean = np.array(data.sample_info['mean'])
        grasp_success = check_succces_with_whole_gewa_dataset(grasp_pred,  0.03, np.deg2rad(30), grasp_gt_path, pointcloud_mean)
        # grasp_success = check_succces_with_whole_dataset(grasp_pred, grasp_dict, 0.03, np.deg2rad(30))
        print(f"Grasp success rate: {grasp_success}")
        success_rates.append(grasp_success)

    #average success rate
    success_rates = np.array(success_rates)
    print(f"Average success rate: {np.mean(success_rates)}")

    #plot the histogram of success rates
    import matplotlib.pyplot as plt
    plt.hist(success_rates)
    plt.show()

    #save the success rates
    np.save("valid_success_rates.npy", success_rates)
    low_success_rates = success_rates[success_rates < 0.5]
    print("Low success rates: ", low_success_rates)
    # # Display the result
    # pair_scores = data.pair_scores.numpy()
    # pos = data.pos.numpy()  
    # pred_pair_scores = pair_classification_pred.squeeze().detach().numpy()
    

    # print(f"min: {np.min(pred_pair_scores)}, max: {np.max(pred_pair_scores)}")
    # threshold = 0.5
    # num_pred_pairs = np.sum(pred_pair_scores > threshold)
    # print(f"Number of predicted pairs: {num_pred_pairs}/{len(pred_pair_scores)}")

    # num_gt_pairs = np.sum(pair_scores > 0)
    # print(f"Number of ground truth pairs: {num_gt_pairs}/{len(pair_scores)}")
    
    # view_params = {
    # 'zoom': 0.8,
    # 'front': [1, 0.5, 1],    # Camera direction
    # 'up': [0, 0, 1]         # Up direction
    # }
    
    # show_pair_edges(pos, pred_pair_scores, dataset.triu_indices, threshold=threshold, view_params=view_params)
    # show_pair_edges(pos, pair_scores, dataset.triu_indices, view_params=view_params)

    # #display grasps
    # print(selected_edge_idxs.shape)
    # grasp_pred = grasp_pred.reshape(selected_edge_idxs.shape[1], 16)

    # sample_info = data.sample_info
    # selected_edge_idxs = selected_edge_idxs.squeeze()
    # pointcloud_mean = pointcloud_mean.reshape(-1)
    # show_grasp_and_edge_predictions(pos, grasp_dict, selected_edge_idxs, grasp_pred,
    #                                 dataset.triu_indices, sample_info, mean=pointcloud_mean, num_grasp_to_show=50, 
    #                                 view_params=view_params)
    