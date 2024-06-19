import torch
# from torch_geometric.nn import PointNetConv
import torch.nn as nn
import torch.nn.functional as F
from PointNet2 import GlobalSAModule, SAModule

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, knn_interpolate


class FPModule(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x, pos_skip, batch_skip
    
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x, pos, edge_index):
        encoded = self.encoder(x, pos, edge_index)
        decoded = self.decoder(*encoded)
        return decoded
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
    
    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        return sa0_out, sa1_out, sa2_out, sa3_out

class Decoder(nn.Module):
    def __init__(self, num_classes=3):
        super(Decoder, self).__init__()
        
        self.fp3_module = FPModule(1, MLP([1024 + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.mlp = MLP([128, 128, 128, num_classes], dropout=0.5, norm=None)
    
    def forward(self, sa0_out, sa1_out, sa2_out, sa3_out):
        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        x = self.mlp(x)
        return x

if __name__ == "__main__":
    model = Autoencoder() 
    num_meshes = 2
    mesh_vertex_count = 50
    total_vertex_count = num_meshes * mesh_vertex_count
    data = torch.randn((total_vertex_count, 3))
    pos = torch.randn((total_vertex_count, 3))
    batch = torch.arange(total_vertex_count, dtype=torch.long)
    output = model(pos, pos, batch)
    print(output.shape)