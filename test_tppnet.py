import torch
import torch.nn as nn
from torch_geometric.data import Data
# from GraspNet import GraspNet
from TppNet import TppNet
from tpp_dataset import TPPDataset
from visualize_tpp_dataset import show_pair_edges
import argparse
import wandb
from torch_geometric.loader import DataListLoader, DataLoader
import numpy as np
from torcheval.metrics.functional.classification import binary_recall
# from sklearn.metrics import recall_score
from metrics import count_correct_approach_scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    parser.add_argument('-i', '--sample_idx', type=int, default=0)
    args = parser.parse_args()
    model_name = args.model_name

    # Initialize a run dont upload the run info

    run = wandb.init(project="Grasp", job_type="download_model", notes="inference")

    # idx 17
    # Access and download model. Returns path to downloaded artifact
    # downloaded_model_path = run.use_model(name="TppNet_nm_500__bs_32.pth_epoch_130_acc_0.962_recall_0.509_prec_0.270.pth:v0")
    #contrastive loss
    # downloaded_model_path = run.use_model(name="TppNet_nm_2000__bs_32.pth_epoch_450_acc_0.972_recall_0.517_prec_0.400.pth:v0") 
    # downloaded_model_path = run.use_model(name="TppNet_nm_2000__bs_32.pth_epoch_390_acc_0.970_recall_0.574_prec_0.381.pth:v0")
    # wo contrastive loss
    downloaded_model_path = run.use_model(name="TppNet_nm_2000__bs_32.pth_epoch_400_acc_0.972_recall_0.523_prec_0.413.pth:v0")
    print(downloaded_model_path)

    model_path = downloaded_model_path

    # load the GraspNet model and run inference then display the gripper pose
    model = TppNet()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    from create_tpp_dataset import save_split_samples
    train_paths, val_paths = save_split_samples('../data', 200)
    # from load_tpp_samples import save_split_samples
    # train_paths, val_paths = save_split_samples('../data', 5)

    dataset = TPPDataset(val_paths, return_pair_dict=True)
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
    print(data.sample_info)
    pair_classification_pred, pair_dot_product, _, _ = model(data)

    test_pair_accuracy = count_correct_approach_scores(pair_dot_product, data.pair_scores)
    test_pair_accuracy = test_pair_accuracy / len(data.pair_scores)
    print(f"Test pair accuracy: {test_pair_accuracy}")

    pair_classification_pred = torch.flatten(pair_classification_pred)
    binary_pair_scores_gt = torch.flatten(data.pair_scores).int()
    test_recall = binary_recall(pair_classification_pred, binary_pair_scores_gt)
    print(f"Test pair recall: {test_recall}")

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