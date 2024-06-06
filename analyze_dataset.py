from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from utils import plot_point_cloud, plot_network
from acronym_dataset import AcronymDataset

if __name__ == "__main__":
    # dataset = ShapeNet(root='data/ShapeNet', categories=['Mug'], split='val', pre_transform=T.KNNGraph(k=6))
    dataset = AcronymDataset('train_success_simplified_acronym_samples.npy')

    # plot_point_cloud(dataset[0])
    for sample in enumerate(dataset):
        print(sample)
        break