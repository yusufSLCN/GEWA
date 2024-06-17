import torch
from torch_geometric.data import Data
from GraspNet import GraspNet
from acronym_dataset import AcronymDataset
from acronym_visualize_output import visualize_grasp
import argparse
import wandb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    parser.add_argument('-idx', '--sample_idx', type=int, default=0)
    args = parser.parse_args()
    model_name = args.model_name

    # Initialize a run
    run = wandb.init(project="GEWA", job_type="download_model")


    # Access and download model. Returns path to downloaded artifact
    downloaded_model_path = run.use_model(name="GraspNet_nm_4000__bs_64_epoch_40.pth:v0")
    print(downloaded_model_path)

    model_path = downloaded_model_path

    # load the GraspNet model and run inference then display the gripper pose
    model = GraspNet(scene_feat_dim= 1028, predictor_out_size=16)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    dataset = AcronymDataset('sample_dirs/valid_success_simplified_acronym_meshes.npy')
    samlpe_idx = args.sample_idx
    data = dataset[samlpe_idx]
    pos = data[0]
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    querry_point = data[2]['query_point'].unsqueeze(0)
    print(querry_point.shape)
    print(pos.shape)
    model.eval()
    output = model(None, pos, batch, querry_point)
    grasp = output.detach().numpy().reshape(4, 4)
    print(grasp)
    visualize_grasp(data[0].numpy(), grasp, data[2]['query_point'].numpy())
