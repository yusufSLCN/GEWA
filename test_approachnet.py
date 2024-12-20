import torch
import torch.nn as nn
from torch_geometric.data import Data
# from GraspNet import GraspNet
from models.ApproachNet import ApproachNet
from dataset.approach_dataset import ApproachDataset
from dataset.create_approach_dataset import save_contactnet_split_samples
from utils.visualize_acronym_dataset import visualize_grasp, visualize_gt_and_pred_gasps
import argparse
import wandb
from torch_geometric.loader import DataLoader
from torcheval.metrics.functional.classification import binary_recall, binary_precision, binary_accuracy
from utils.metrics import check_succces_with_whole_gewa_dataset
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-idx', '--sample_idx', type=int, default=0)
    args = parser.parse_args()

    # Initialize a run dont upload the run info
    run = wandb.init(project="Grasp", job_type="download_model", notes="inference")

    # idx 17
    # Access and download model. Returns path to downloaded artifact
    # downloaded_model_path = run.use_model(name="DynANet_nm_1000__bs_64_epoch_820.pth:v0")
    # downloaded_model_path = run.use_model(name="DynANet_nm_1000__bs_128_epoch_900.pth:v0")
    # downloaded_model_path = run.use_model(name="DynANet_nm_4000__bs_128_epoch_2020.pth:v0")
    downloaded_model_path = run.use_model(name="DynANet_nm_1200__bs_64__gd_9__gs_50.pth_epoch_1370_grasp_success_0.5898125000000001.pth:v0")
    print(downloaded_model_path)

    model_path = downloaded_model_path

    # load the GraspNet model and run inference then display the gripper pose
    num_grasp_samples = 10
    model = ApproachNet(grasp_dim=9, num_grasp_sample=num_grasp_samples, sort_by_score=True)
    #count model parameters
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # train_paths, val_paths = save_split_samples('../data', -1)
    train_paths, val_paths = save_contactnet_split_samples('../data', num_mesh=1200)
    dataset = ApproachDataset(val_paths)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    samlpe_idx = args.sample_idx
    # data = data_loader[samlpe_idx]
    config = wandb.config
    config.model_path = model_path
    config.model = model.__class__.__name__
    config.sample_idx = samlpe_idx
    config.num_samples = len(data_loader)

    # print(data)
    model.device = 'cpu'
    model.eval()

    for i, d in enumerate(data_loader):
        data = d
        if i == samlpe_idx:
            break
    print(data.sample_info)
        
    approach_score_pred, grasp_pred, approach_points, grasp_gt, num_grasps_of_approach_points = model(data)
    # grasp_pred[3, :] = [0, 0, 0, 1]
    binary_approach_score_gt = (data[0].approach_scores > 0).int()
    binary_approach_score_pred = (approach_score_pred > 0.5).int()

    print(approach_score_pred.shape, binary_approach_score_gt.shape)
    approach_acc = binary_accuracy(binary_approach_score_pred, binary_approach_score_gt)
    approach_recall = binary_recall(binary_approach_score_pred, binary_approach_score_gt)
    approach_precision = binary_precision(binary_approach_score_pred, binary_approach_score_gt)


    pred = grasp_pred.cpu().detach().reshape(-1, 1, 4, 4).numpy()

    gt = grasp_gt.cpu().detach().reshape(-1, num_grasp_samples, dataset.max_grasp_perpoint, 4, 4).numpy()
    num_grasps_of_approach_points = num_grasps_of_approach_points.cpu().detach().reshape(-1, num_grasp_samples).numpy()
    # grasps_success = check_batch_grasp_success_rate_per_point(pred, gt, 0.03, np.deg2rad(30), num_grasps_of_approach_points)
    grasp_gt_path = data.sample_info['grasps'][0]
    point_cloud_mean = data.sample_info['mean'].numpy()
    grasp_success = check_succces_with_whole_gewa_dataset(pred, 0.03, np.deg2rad(30), grasp_gt_path, point_cloud_mean)
    print(f"Approach accuracy: {approach_acc}, recall: {approach_recall}, precision: {approach_precision}")
    print(f"Grasp success rate: {grasp_success}")

    # visualize_grasp(data[0].numpy(), grasp, data[2]['query_point'].numpy())
    print(approach_score_pred.shape, pred.shape, approach_points.shape, gt.shape, num_grasps_of_approach_points.shape)
    num_of_grasps = 3
    grasp_pred = pred[:num_of_grasps, 0].reshape(-1, 4, 4)
    grasp_gt = gt[0, :num_of_grasps, 0].reshape(-1, 4, 4)
    approach_points = approach_points[:num_of_grasps].detach().numpy()
    approach_score_pred = (approach_score_pred > 0.5).float().numpy()
    approach_score_gt = (data[0].approach_scores > 0).float().numpy()
    num_grasps_of_approach_points = num_grasps_of_approach_points.flatten()
    num_grasps_of_approach_points = num_grasps_of_approach_points[:num_of_grasps]
    print(grasp_gt.shape, grasp_pred.shape, approach_points.shape, approach_score_pred.shape)
    # visualize_gt_and_pred_gasps(data[0].pos.numpy(), grasp_gt, grasp_pred, approach_points, approach_score_gt, num_grasps_of_approach_points)
    visualize_gt_and_pred_gasps(data[0].pos.numpy(), grasp_gt, grasp_pred, approach_points, approach_score_pred, num_grasps_of_approach_points)