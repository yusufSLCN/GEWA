import torch
from torch_geometric.data import Data
# from GraspNet import GraspNet
from EdgeGraspNet import EdgeGraspNet
from acronym_dataset import AcronymDataset
from acronym_visualize_dataset import visualize_grasp, visualize_gt_and_pred_gasp
import argparse
import wandb

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-mn','--model_name', type=str, default='model')
    parser.add_argument('-idx', '--sample_idx', type=int, default=0)
    args = parser.parse_args()
    model_name = args.model_name

    # Initialize a run dont upload the run info

    run = wandb.init(project="GEWA", job_type="download_model", notes="inference")


    # Access and download model. Returns path to downloaded artifact
    # downloaded_model_path = run.use_model(name="GraspNet_nm_4000__bs_64_epoch_40.pth:v0")
    downloaded_model_path = run.use_model(name="GraspNet_nm_1000__bs_64_epoch_200.pth:v2")
    print(downloaded_model_path)

    model_path = downloaded_model_path

    # load the GraspNet model and run inference then display the gripper pose
    model = EdgeGraspNet(scene_feat_dim=1028)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    dataset = AcronymDataset('sample_dirs/valid_success_simplified_acronym_meshes.npy')
    samlpe_idx = args.sample_idx
    data = dataset[samlpe_idx]
    pos = data[0]
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    query_point_idx = data[2]['query_point_idx']
    query_point_idx = torch.tensor(query_point_idx, dtype=torch.int64).unsqueeze(0)
    print(query_point_idx)
    print(pos.shape)
    model.eval()
    output = model(None, pos, batch, query_point_idx)

    pred_grasp = output.detach().numpy().reshape(4, 4)
    print(pred_grasp)
    # visualize_grasp(data[0].numpy(), grasp, data[2]['query_point'].numpy())
    gt_grasp = data[1].reshape(4, 4)
    visualize_gt_and_pred_gasp(data[0].numpy(), gt_grasp.numpy(), pred_grasp, data[2]['query_point'].numpy())
