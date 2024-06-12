from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from utils import plot_point_cloud, plot_network
from acronym_dataset import AcronymDataset
import numpy as np

if __name__ == "__main__":
    # dataset = ShapeNet(root='data/ShapeNet', categories=['Mug'], split='val', pre_transform=T.KNNGraph(k=6))
    dataset = AcronymDataset('sample_dirs/train_success_simplified_acronym_samples.npy')

    # plot_point_cloud(dataset[0])
    print(dataset[2][0].shape)
    print(dataset[2][1].shape)
    #calculate min, max, mean and std of the vertex counts and the center locations
    vertex_counts = []
    center_locations = []
    for i in range(len(dataset)):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(dataset)}")

        sample = dataset[i]
        vertices = sample[0]
        vertex_counts.append(vertices.shape[0])
        center = vertices.mean(0)
        center_locations.append(center)
    
    vertex_counts = np.array(vertex_counts)
    center_locations = np.array(center_locations)
    print(f"Min vertex count: {vertex_counts.min()}")
    print(f"Max vertex count: {vertex_counts.max()}")
    print(f"Mean vertex count: {vertex_counts.mean()}")
    print(f"Std vertex count: {vertex_counts.std()}")
    print(f"Min center location: {center_locations.min(0)}")
    print(f"Max center location: {center_locations.max(0)}")
    print(f"Mean center location: {center_locations.mean(0)}")

