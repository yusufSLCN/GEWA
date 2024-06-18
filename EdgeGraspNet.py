import torch
# from torch_geometric.nn import PointNetConv
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius
from GraspNet import SAModule, GlobalSAModule

class EdgeGraspPredictor(nn.Module):
    def __init__(self, scene_feat_dim, point_feat_dim=256, out_size=9):
        super(EdgeGraspPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(scene_feat_dim + point_feat_dim, scene_feat_dim  // 2),
            nn.ReLU(),
            nn.Linear(scene_feat_dim // 2, scene_feat_dim // 4),
            nn.ReLU(),
            nn.Linear(scene_feat_dim // 4, out_size),
        )
    def forward(self, edge_features, query_point_idx):
        # expanded_querry_point_feat = querry_point_feat[point_batch]
        # h = torch.cat([edge_features, expanded_querry_point_feat], dim=1)
        edge_feat = edge_features[query_point_idx]
        h = self.predictor(edge_feat)
        return h

class Encoder(torch.nn.Module):
    def __init__(self, out_channels, point_feat_dim=256, ratios=[1.0, 1.0]):
        super().__init__()

        self.sa1_module = SAModule(ratios[0], 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(ratios[1], 0.4, MLP([128 + 3, 128, 128, point_feat_dim]))
        self.sa3_module = GlobalSAModule(MLP([point_feat_dim + 3, point_feat_dim, 512, out_channels]))

    def forward(self, x, pos, batch):
        sa1_out = self.sa1_module(x, pos, batch)
        sa2_out = self.sa2_module(*sa1_out)
        point_feat, _, point_batch = sa2_out
        sa3_out = self.sa3_module(*sa2_out)
        # dublicate the scene features to match the number of points in the batch
        expanded_scene_feat = sa3_out[0][point_batch]
        # create edge features by concatenating the scene features with the point features 
        edge_feat = torch.cat([point_feat, expanded_scene_feat], dim=1)
        x, pos, batch = sa3_out

        return x, edge_feat, point_batch
    
class EdgeGraspNet(nn.Module):
    def __init__(self, scene_feat_dim=1024, point_feat_dim=256, predictor_out_size = 9):
        super(EdgeGraspNet, self).__init__()
        self.encoder = Encoder(out_channels = scene_feat_dim, point_feat_dim=point_feat_dim)
        self.predictor = EdgeGraspPredictor(scene_feat_dim, point_feat_dim=point_feat_dim, out_size=predictor_out_size)
    def forward(self, x, pos, batch, query_point_idx):
        x, edge_feat, point_batch = self.encoder(x, pos, batch)
        output = self.predictor(edge_feat, query_point_idx)
        trans_m = self.calculateTransformationMatrix(output)
        grasp = trans_m.view(-1, 16)
        return grasp
    
    def calculateTransformationMatrix(self, grasp):
        translation = grasp[:, :3]
        r1 = grasp[:, 3:6]
        r2 = grasp[:, 6:]
        #orthogonalize the rotation vectors
        r1 = r1 / torch.norm(r1, dim=1, keepdim=True)
        r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
        r2 = r2 / torch.norm(r2, dim=1, keepdim=True)
        r3 = torch.cross(r1, r2, dim=1)
        #create the rotation matrix
        r = torch.stack([r1, r2, r3], dim=2)
        #create 4x4 transformation matrix for each 
        trans_m = torch.eye(4).repeat(len(grasp), 1, 1).to(grasp.device)
        trans_m[:,:3, :3] = r
        trans_m[:, :3, 3] = translation
        return trans_m
    

if __name__ == "__main__":
    model = EdgeGraspNet(scene_feat_dim= 2048, point_feat_dim=256, predictor_out_size=9) 

    data = torch.randn((50, 3))
    pos = torch.randn((50, 3))
    batch = torch.arange(50, dtype=torch.long)
    print(pos.shape)
    for i in range(10):
        batch[i*5:(i+1)*5] = i
    query_point_idx = torch.zeros(10, dtype=torch.long)
    grasps = model(None, pos, batch, query_point_idx)
    print(grasps.shape)