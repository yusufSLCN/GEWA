import torch
import torch.nn as nn
from torch_geometric.data import Data
# from GraspNet import GraspNet
from TppNet import TppNet
from tpp_dataset import TPPDataset
from visualize_tpp_dataset import show_pair_edges, show_grasp_and_edge_predictions
import argparse
import wandb
from torch_geometric.loader import DataListLoader, DataLoader
import numpy as np
from torcheval.metrics.functional.classification import binary_recall, binary_precision, binary_accuracy
from metrics import check_succces_with_whole_dataset
from create_tpp_dataset import save_split_samples


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    parser.add_argument('-i', '--sample_idx', type=int, default=0)
    args = parser.parse_args()
    model_name = args.model_name

    # Initialize a run dont upload the run info

    run = wandb.init(project="Grasp", job_type="eval", notes="validation")

    # idx 4, 5, 6, 7, 13, 18
    # with grasp head
    # downloaded_model_path = run.use_model(name="TppNet_nm_1000__bs_8.pth_epoch_90_acc_0.93_recall_0.62.pth:v0")

    #500 * tiploss
    # downloaded_model_path = run.use_model(name="TppNet_nm_1000__bs_8.pth_epoch_90_acc_0.93_recall_0.57.pth:v0")

    #100 tip loss + axis loss 
    downloaded_model_path = run.use_model(name="TppNet_nm_1000__bs_8.pth_epoch_540_acc_0.97_recall_0.42.pth:v0")

    # 200 tip loss + axis loss
    # downloaded_model_path = run.use_model(name="TppNet_nm_1000__bs_8.pth_epoch_210_acc_0.97_recall_0.33.pth:v0")
    
    #wo tip loss 
    # downloaded_model_path = run.use_model(name="TppNet_nm_100__bs_4.pth_epoch_950_acc_0.94_recall_0.56.pth:v0")
    print(downloaded_model_path)

    model_path = downloaded_model_path


    # load the GraspNet model and run inference then display the gripper pose
    model = TppNet()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    train_paths, val_paths = save_split_samples('../data', 400, dataset_name="tpp_effdict")

    dataset = TPPDataset(val_paths, return_pair_dict=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    config = wandb.config
    config.model_path = model_path
    config.model = model.__class__.__name__
    config.dataset = dataset.__class__.__name__
    config.valid_dataset_size = len(dataset)


    # print(data)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()
    average_recall = 0
    average_precision = 0
    average_pair_accuracy = 0
    average_grasp_success = 0

    for i, data in enumerate(data_loader):
        print(f"Sample {i}/{len(data_loader)}")
        if device == torch.device("cuda"):
            data.pos = data.pos.to(device)
            data.batch = data.batch.to(device)
            
        grasp_pred, selected_edge_idxs, mid_edge_pos, _, grasp_target, num_valid_grasps, pair_classification_pred, pair_dot_product = model(data)
        # pair_classification_pred, pair_dot_product, _, _ = model(data)


        pair_classification_pred = torch.flatten(pair_classification_pred)
        binary_pair_scores_gt = torch.flatten(data.pair_scores).int()
        val_recall = binary_recall(pair_classification_pred, binary_pair_scores_gt)
        val_precision = binary_precision(pair_classification_pred, binary_pair_scores_gt)
        val_pair_accuracy = binary_accuracy(pair_classification_pred, binary_pair_scores_gt)


        grasp_dict = data.y[0][0]
        grasp_pred = grasp_pred.detach().numpy()
        grasp_pred = grasp_pred.reshape( -1, 1, 4, 4)
        grasp_success = check_succces_with_whole_dataset(grasp_pred, grasp_dict, 0.03, np.deg2rad(30))
        print(f"Grasp success rate: {grasp_success}")

        average_recall += val_recall
        average_precision += val_precision
        average_pair_accuracy += val_pair_accuracy
        average_grasp_success += grasp_success
    
    average_recall = average_recall / len(data_loader)
    average_precision = average_precision / len(data_loader)
    average_pair_accuracy = average_pair_accuracy / len(data_loader)
    average_grasp_success = average_grasp_success / len(data_loader)
    wandb.log({"Average Recall": average_recall, "Average Precision": average_precision, 
               "Average Pair Accuracy": average_pair_accuracy, "Average Grasp Success": average_grasp_success})
    run.finish()

    print(f"Average Recall: {average_recall:.2f}, Average Precision: {average_precision:.2f}, Average Pair Accuracy: {average_pair_accuracy:.2f}, Average Grasp Success: {average_grasp_success:.2f}")
