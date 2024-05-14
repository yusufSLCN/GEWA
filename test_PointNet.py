import torch
from torch_geometric.data import Data
from PointNet import PointNet

# Create a PointNet model
model = PointNet()

# Create a test input
test_pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
test_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
test_batch = torch.tensor([0, 0, 1])

# Pass the test input through the model
output = model(test_pos, test_edge_index, test_batch)

# Print the output
print(output)