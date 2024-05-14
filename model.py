import torch
# from torch_geometric.nn import PointNetConv
from PointNetLayer import PointNetLayer
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, num_points, num_features):
        super(Autoencoder, self).__init__()
        
        self.encoder = Encoder(num_points, num_features)
        self.decoder = Decoder(num_features, num_points)
    
    def forward(self, x, pos, edge_index):
        encoded = self.encoder(x, pos, edge_index)
        decoded = self.decoder(encoded)
        return decoded
    
class Encoder(nn.Module):
    def __init__(self, num_points, num_features):
        super(Encoder, self).__init__()
        
        self.conv1 = PointNetLayer(num_points, num_features)
        self.conv2 = PointNetLayer(num_features, num_features)
    
    def forward(self, x, pos, edge_index):
        h = self.conv1(x, pos, edge_index)
        h = h.relu()
        h = self.conv2(h, pos, edge_index)
        h = h.relu()
        return h

class Decoder(nn.Module):
    def __init__(self, num_features, num_points):
        super(Decoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_points)
        )
    
    def forward(self, x):
        decoded = self.decoder(x)
        return decoded

if __name__ == "__main__":
    model = Autoencoder(3, 32) 

    test_pos = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 1.0, 1.1], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [1.0, 1.1, 1.2], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0], [1.1, 1.2, 1.3]]) 
    test_edge_index = torch.tensor([[0, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0]])
    output = model(test_pos, test_pos, test_edge_index)
    print(output.shape)