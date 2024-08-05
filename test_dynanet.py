import torch
import torch.nn as nn
from torch_geometric.data import Data
# from GraspNet import GraspNet
from DynANet import DynANet
from gewa_dataset import GewaDataset
from create_gewa_dataset import save_split_samples
from acronym_visualize_dataset import visualize_grasp, visualize_gt_and_pred_gasps
import argparse
import wandb
from torch_geometric.loader import DataListLoader, DataLoader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    parser.add_argument('-idx', '--sample_idx', type=int, default=0)
    args = parser.parse_args()
    model_name = args.model_name

    # Initialize a run dont upload the run info

    run = wandb.init(project="Grasp", job_type="download_model", notes="inference")

    # idx 17
    # Access and download model. Returns path to downloaded artifact
    # downloaded_model_path = run.use_model(name="DynANet_nm_1000__bs_64_epoch_820.pth:v0")
    # downloaded_model_path = run.use_model(name="DynANet_nm_1000__bs_128_epoch_900.pth:v0")
    downloaded_model_path = run.use_model(name="DynANet_nm_4000__bs_128_epoch_2020.pth:v0")
    print(downloaded_model_path)

    model_path = downloaded_model_path

    # load the GraspNet model and run inference then display the gripper pose
    model = DynANet(grasp_dim=9, num_grasp_sample=400)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    train_paths, val_paths = save_split_samples('../data', -1)
    dataset = GewaDataset(val_paths)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    samlpe_idx = args.sample_idx
    # data = data_loader[samlpe_idx]

    # print(data)
    model.device = 'cpu'
    model.eval()

    for i, d in enumerate(data_loader):
        data = d
        if i == samlpe_idx:
            break

    approach_score_pred, grasp_pred, approach_points, grasp_gt, num_grasps_of_approach_points = model(data)
    # grasp_pred[3, :] = [0, 0, 0, 1]

    # visualize_grasp(data[0].numpy(), grasp, data[2]['query_point'].numpy())
    num_of_grasps = 5
    grasp_pred = grasp_pred[:num_of_grasps].detach().numpy().reshape(-1, 4, 4)
    grasp_gt = grasp_gt[0, :num_of_grasps, 0].detach().numpy().reshape(-1, 4, 4)
    approach_points = approach_points[:num_of_grasps].detach().numpy()
    approach_score_pred = (approach_score_pred > 0.5).float().numpy()
    approach_score_gt = (data[0].approach_scores > 0).float().numpy()
    num_grasps_of_approach_points = num_grasps_of_approach_points[:num_of_grasps].detach().numpy()
    print(grasp_gt.shape, grasp_pred.shape, approach_points.shape, approach_score_pred.shape)
    # visualize_gt_and_pred_gasps(data[0].pos.numpy(), grasp_gt, grasp_pred, approach_points, approach_score_gt, num_grasps_of_approach_points)
    visualize_gt_and_pred_gasps(data[0].pos.numpy(), grasp_gt, grasp_pred, approach_points, approach_score_pred, num_grasps_of_approach_points)
