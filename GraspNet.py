import torch
# from torch_geometric.nn import PointNetConv
import torch.nn as nn
from PointNet2 import Encoder


class GraspPredictor(nn.Module):
    def __init__(self, enc_out_channels, predictor_out_size):
        super(GraspPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(enc_out_channels, enc_out_channels // 2),
            nn.ReLU(),
            nn.Linear(enc_out_channels // 2, enc_out_channels // 4),
            nn.ReLU(),
            nn.Linear(enc_out_channels // 4, predictor_out_size),
        )
    
    def forward(self, x):
        h = self.predictor(x)
        return h

class GraspEvaluator(nn.Module):
    def __init__(self, enc_out_channels, predictor_out_size):
        super(GraspEvaluator, self).__init__()

        self.grasp_encoder = nn.Sequential(
            nn.Linear(predictor_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, enc_out_channels),
        )
        self.evaluator = nn.Sequential(
            nn.Linear(enc_out_channels * 2, enc_out_channels),
            nn.ReLU(),
            nn.Linear(enc_out_channels, enc_out_channels // 2),
            nn.ReLU(),
            nn.Linear(enc_out_channels // 2, enc_out_channels // 4),
            nn.ReLU(),
            nn.Linear(enc_out_channels // 4, 1),
        )
    
    def forward(self, scene_embed, grasp_pose):
        grasp_feat = self.grasp_encoder(grasp_pose)
        h = torch.cat([scene_embed, grasp_feat], dim=1)
        grasp_score = self.evaluator(h)
        return grasp_score

class GraspNet(nn.Module):
    def __init__(self, enc_out_channels, predictor_out_size):
        super(GraspNet, self).__init__()
        
        self.encoder = Encoder(enc_out_channels)
        self.predictor = GraspPredictor(enc_out_channels, predictor_out_size)
        # self.evaluator = GraspEvaluator(enc_out_channels, predictor_out_size)
    
    def forward(self, x, pos, batch):
        scene_feat = self.encoder(x, pos, batch)
        grasp = self.predictor(scene_feat)
        # grasp_score = self.evaluator(scene_feat, grasp)
        return grasp

if __name__ == "__main__":
    model = GraspNet(enc_out_channels= 1028, predictor_out_size=16) 

    data = torch.randn((10, 50, 3))
    pos = torch.randn((10, 50, 3))
    batch = torch.zeros(10, dtype=torch.long)
    print(pos.shape)
    grasp = model(None, pos, batch)
    print(grasp.shape)