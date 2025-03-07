from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
import torch
import torch.nn.functional as F


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super().__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch

class Encoder(torch.nn.Module):
    def __init__(self, out_channels=1024, ratios=[0.5, 0.25]):
        super().__init__()

        self.sa1_module = SAModule(ratios[0], 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(ratios[1], 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, out_channels]))

    def forward(self, x, pos, batch):
        sa1_out = self.sa1_module(x, pos, batch)
        sa2_out = self.sa2_module(*sa1_out)
        # print(f"{sa2_out[0].shape=}")
        sa3_out = self.sa3_module(*sa2_out)
        # print(f"{sa3_out[0].shape=}")

        x, pos, batch = sa3_out

        return x

class PointNet2Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.encoder = Encoder(out_channels = 1024)
        self.mlp = MLP([1024, 512, 256, 10], dropout=0.5, norm=None)

    def forward(self, x, pos, batch):
        x = self.encoder(x, pos, batch)
        return self.mlp(x).log_softmax(dim=-1)
    

if __name__ == "__main__":
    # run the net model with random input
    model = PointNet2Classifier()
    data = torch.randn((50, 3))
    pos = torch.randn((50, 3))
    batch = torch.zeros(50, dtype=torch.long)
    output = model(None, pos, batch)
    print(output.shape)

