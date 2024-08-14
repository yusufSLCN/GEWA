import torch
import torch.nn as nn
from torch_geometric.data import Data
# from GraspNet import GraspNet
from TppNet import TppNet
from tpp_dataset import TPPDataset
from create_tpp_dataset import save_split_samples
from visualize_tpp_dataset import show_pair_edges
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
    downloaded_model_path = run.use_model(name="TppNet_nm_100__bs_64.pth_epoch_285.pth:v0")
    print(downloaded_model_path)

    model_path = downloaded_model_path

    # load the GraspNet model and run inference then display the gripper pose
    model = TppNet()
    # model = nn.DataParallel(model)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    train_paths, val_paths = save_split_samples('../data', 100)
    dataset = TppNet(val_paths)
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

    pair_classification_pred, pair_dot_product = model(data)
    print(pair_dot_product.shape)