import torch
from torch_geometric.data import Data
from GraspNet import GraspNet
from acronym_dataset import AcronymDataset
from acronym_visualize_output import visualize_grasp
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    args = parser.parse_args()
    model_name = args.model_name
    model_path = f'models/{model_name}.pth'

    # load the GraspNet model and run inference then display the gripper pose
    model = GraspNet(enc_out_channels= 1028, predictor_out_size=16)
    model.load_state_dict(torch.load(model_path))
    dataset = AcronymDataset('sample_dirs/valid_success_simplified_acronym_meshes.npy')
    data = dataset[0]
    x = data[0].unsqueeze(0)
    pos = torch.zeros((x.shape[0], 3))
    batch = torch.zeros(x.shape[0], dtype=torch.long)
    querry_point = data[2]['query_point'].unsqueeze(0)
    print(querry_point.shape)
    model.eval()
    output = model(x, pos, batch, querry_point)
    grasp = output.detach().numpy().reshape(4, 4)
    visualize_grasp(data[0].numpy(), grasp)
