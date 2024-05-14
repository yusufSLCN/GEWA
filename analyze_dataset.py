from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from utils import plot_point_cloud, plot_network

if __name__ == "__main__":
    dataset = ShapeNet(root='data/ShapeNet', categories=['Mug'], split='val', pre_transform=T.KNNGraph(k=6))

    # plot_point_cloud(dataset[0])
    plot_network(dataset[0])