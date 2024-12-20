import torch
import numpy as np
from torch_geometric.nn import DynamicEdgeConv, MLP, radius

# arr = np.arange(32).reshape(-1, 4, 4)
# print(arr)
# arr =  arr.reshape(-1, 16)
# print(arr)
# t_arr = torch.arange(32).reshape(-1, 4, 4)
# print(t_arr)
# t_arr = t_arr.reshape(-1, 16)
# print(t_arr)

# pair1 = set((1,3))
# pair2 = set((3,1))
# print(pair1 == pair2)
# vertices = np.random.rand(5, 5)
# upper_tri_idx = np.triu_indices(vertices.shape[0], k=1)
# pairs = vertices[upper_tri_idx]
# print(upper_tri_idx)
# print(pairs.shape)

# triu = torch.triu_indices(3, 3, offset=1)
# batched_shared_features = torch.ones(2, 3, 4)
# batched_shared_features[:, 1, :] = 2
# batched_shared_features[:, 2, :] = 3
# dot_product = torch.matmul(batched_shared_features, batched_shared_features.transpose(1, 2))
# print(dot_product)
# print(dot_product.shape)
# upper = dot_product[:, triu[0], triu[1]]
# print(upper)
# gt = torch.ones(2, 16) * torch.range(0, 15)
# pred = torch.zeros(2, 16)
# import torch.nn as nn
# grasp_mse_loss = nn.MSELoss(reduction='none')
# loss = grasp_mse_loss(pred, gt)
# print(loss.shape)
# print(loss)
# points = torch.zeros((10, 6))
# conv1 = DynamicEdgeConv(MLP([12, 16, 32]), k=8, aggr='max')
# out = conv1(points)
# from create_tpp_dataset import get_contactnet_split
# train_meshes, valid_meshes = get_contactnet_split()
# print(train_meshes[0])

# grasps = np.random.rand(2, 1024, 1, 4, 4)  # Shape (2, 1024, 1, 4, 4)
# indexes = np.array([
#     [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],  # Indices for first group
#     [50, 150, 250, 350, 450, 550, 650, 750, 850, 950]     # Indices for second group
# ])  # Shape (2, 10)

# # Use advanced indexing to select based on indexes
# result = grasps[np.arange(2)[:, None], indexes]  # Shape (2, 10, 1, 4, 4)

# print(result.shape) 
# print(grasps[0, indexes[0]].shape)
# print(result[0].shape)
# #check if they are equal
# print(np.allclose(result[0], grasps[0, indexes[0]]))

points = torch.rand((100, 3)) 
radiuss = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
batch = torch.zeros(5, dtype=torch.long)
# print(points)
cropped_points = radius(points, batch, r= radiuss, max_num_neighbors=10)
print(cropped_points.shape)
print(cropped_points)