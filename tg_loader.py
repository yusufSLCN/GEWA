from acronym_dataset import AcronymDataset
from torch_geometric.loader import DataListLoader
from torch_geometric.nn import DataParallel
from GewaNet import GewaNet
from create_dataset_paths import save_split_meshes
import torch

if __name__ == "__main__":
    train_dataset, val_dataset = save_split_meshes('../data', 100)
    train_dataset = AcronymDataset(train_dataset)
    train_loader = DataListLoader(train_dataset, batch_size=16, shuffle=True)
    model = GewaNet(scene_feat_dim= 1024, point_feat_dim=256, predictor_out_size=9)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model.multi_gpu = True
        model = DataParallel(model)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for data in train_loader:
        # print(len(data))
        # print(data[0][-1])
        grasp_pred = model(data)
        break