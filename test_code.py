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
a = np.zeros((8, 500, 1, 4, 4))
rep_count = np.arange(5)

rep = np.repeat(a, rep_count, axis=2)
# rep = np.repeat(a, 16, axis=2)
print(rep.shape)
# gt = torch.ones(2, 16) * torch.range(0, 15)
# pred = torch.zeros(2, 16)
# import torch.nn as nn
# grasp_mse_loss = nn.MSELoss(reduction='none')
# loss = grasp_mse_loss(pred, gt)
# print(loss.shape)
# print(loss)
