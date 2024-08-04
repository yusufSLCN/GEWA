import torch
import numpy as np


# arr = np.arange(32).reshape(-1, 4, 4)
# print(arr)
# arr =  arr.reshape(-1, 16)
# print(arr)
# t_arr = torch.arange(32).reshape(-1, 4, 4)
# print(t_arr)
# t_arr = t_arr.reshape(-1, 16)
# print(t_arr)

pair1 = set((1,3))
pair2 = set((3,1))
print(pair1 == pair2)
vertices = np.random.rand(3, 3)
upper_tri_idx = np.triu_indices(vertices.shape[0], k=1)
pairs = vertices[upper_tri_idx]
print(upper_tri_idx)
print(pairs.shape)
# gt = torch.ones(2, 16) * torch.range(0, 15)
# pred = torch.zeros(2, 16)
# import torch.nn as nn
# grasp_mse_loss = nn.MSELoss(reduction='none')
# loss = grasp_mse_loss(pred, gt)
# print(loss.shape)
# print(loss)
