from torch_geometric.nn import global_max_pool
from torch_geometric.nn.conv import PointNetConv
from torch import Tensor
from torch.nn import Linear
import torch
from PointNetLayer import PointNetLayer


from torch_geometric.nn import global_max_pool


class PointNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = PointNetLayer(3, 32)
        self.conv2 = PointNetLayer(32, 32)
        self.classifier = Linear(32, 2)

    def forward(self,
        pos: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:

        # Perform two-layers of message passing:
        h = self.conv1(pos, pos, edge_index)
        h = h.relu()
        h = self.conv2(h, pos, edge_index)
        h = h.relu()

        # Global Pooling:
        h = global_max_pool(h, batch)  # [num_examples, hidden_channels]

        # Classifier:
        return self.classifier(h)




model = PointNet()
# Create a test input
test_pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
test_edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
test_batch = torch.tensor([0, 0, 1])

# Pass the test input through the model
output = model(test_pos, test_edge_index, test_batch)