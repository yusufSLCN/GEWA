import torch
import torch.nn as nn
from torch_geometric.data import Data
from DynANet import DynANet
from gewa_dataset import GewaDataset
import argparse
import wandb
from torch_geometric.loader import DataLoader
import numpy as np
from torcheval.metrics.functional.classification import binary_recall, binary_precision, binary_accuracy
from metrics import check_succces_with_whole_gewa_dataset, check_batch_grasp_success_rate_per_point
from create_gewa_dataset import save_contactnet_split_samples


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-gs', '--grasp_samples', type=int, default=10)
    parser.add_argument('-n', '--notes', type=str, default='')
    parser.add_argument('-sbs', '--sort_by_score', action='store_true')
    args = parser.parse_args()

    # Initialize a run dont upload the run info
    sort_by_score = args.sort_by_score
    num_grasp_samples = args.grasp_samples

    if sort_by_score:
        notes = f"top {num_grasp_samples}" + args.notes
    else:
        notes = args.notes
    run = wandb.init(project="Grasp", job_type="eval", notes=f"validation {notes}")
    config = wandb.config

    # downloaded_model_path = run.use_model(name="DynANet_nm_4000__bs_128_epoch_2020.pth:v0")
    #contactnet split
    downloaded_model_path = run.use_model(name="DynANet_nm_1200__bs_64__gd_9__gs_50.pth_epoch_1370_grasp_success_0.5898125000000001.pth:v0")
    print(downloaded_model_path)

    model_path = downloaded_model_path


    # load the GraspNet model and run inference then display the gripper pose
    model = DynANet(grasp_dim=9, num_grasp_sample=num_grasp_samples, sort_by_score=sort_by_score)
    config.model_name = model.__class__.__name__
    model = nn.DataParallel(model, device_ids=[0])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    train_paths, val_paths = save_contactnet_split_samples('../data', num_mesh=1200)

    dataset = GewaDataset(val_paths)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    config.model_path = model_path
    config.dataset = dataset.__class__.__name__
    config.num_mesh = len(dataset)
    config.grasp_samples = num_grasp_samples


    # print(data)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    average_approach_recall = 0
    average_approach_precision = 0
    average_approach_accuracy = 0
    average_grasp_success = 0

    for i, data in enumerate(data_loader):
        print(f"Sample {i}/{len(data_loader)}")
        if device == torch.device("cuda"):
            data.pos = data.pos.to(device)
            data.batch = data.batch.to(device)
            data.y = data.y.to(device)
            data.num_grasps = data.num_grasps.to(device)
            

        approach_score_pred, grasp_pred, approach_points, grasp_gt, num_grasps_of_approach_points = model(data)
        # pair_classification_pred, pair_dot_product, _, _ = model(data)
        binary_approach_score_gt = (data[0].approach_scores > 0).int()
        binary_approach_score_pred = (approach_score_pred > 0.5).int()
        binary_approach_score_pred = binary_approach_score_pred.detach().cpu()
        # print(approach_score_pred.shape, binary_approach_score_gt.shape)
        approach_acc = binary_accuracy(binary_approach_score_pred, binary_approach_score_gt)
        approach_recall = binary_recall(binary_approach_score_pred, binary_approach_score_gt)
        approach_precision = binary_precision(binary_approach_score_pred, binary_approach_score_gt)

        print(f"Approach accuracy: {approach_acc}, recall: {approach_recall}, precision: {approach_precision}")

        pred = grasp_pred.cpu().detach().reshape(-1, num_grasp_samples, 1, 4, 4).numpy()
        # gt = grasp_gt.cpu().detach().reshape(-1, num_grasp_samples, dataset.max_grasp_perpoint, 4, 4).numpy()
        # num_grasps_of_approach_points = num_grasps_of_approach_points.cpu().detach().reshape(-1, num_grasp_samples).numpy()
        # grasp_success = check_batch_grasp_success_rate_per_point(pred, gt, 0.03, np.deg2rad(30), num_grasps_of_approach_points)
        # print(f"Grasp success rate: {grasp_success}")



        # grasp_dict = data.y[0][0]
        # grasp_pred = grasp_pred.detach().numpy()
        # grasp_pred = grasp_pred.reshape( -1, 1, 4, 4)
        # grasp_pred = pred[0, :num_grasp_samples, 0].reshape(-1, 4, 4)
        pred = pred.reshape(-1, 1, 4, 4)
        grasp_gt_path = data.sample_info['grasps'][0]
        point_cloud_mean = data.sample_info['mean'].numpy()
        grasp_success = check_succces_with_whole_gewa_dataset(pred, 0.03, np.deg2rad(30), grasp_gt_path, point_cloud_mean)
        print(f"Grasp success rate: {grasp_success}")

        average_approach_recall += approach_recall
        average_approach_precision += approach_precision
        average_approach_accuracy += approach_acc
        average_grasp_success += grasp_success
    
    average_approach_recall = average_approach_recall / len(data_loader)
    average_approach_precision = average_approach_precision / len(data_loader)
    average_approach_accuracy = average_approach_accuracy / len(data_loader)
    average_grasp_success = average_grasp_success / len(data_loader)
    wandb.log({"Average Approach Recall": average_approach_recall, "Average Approach Precision": average_approach_precision, 
               "Average Approach Accuracy": average_approach_accuracy, "Average Grasp Success": average_grasp_success})
    run.finish()

    print(f"Average Approach Recall: {average_approach_recall:.2f}, Average Approach Precision: {average_approach_precision:.2f}, Average Approach Accuracy: {average_approach_accuracy:.2f}, Average Grasp Success: {average_grasp_success:.2f}")