import torch
# from torch_geometric.nn import PointNetConv
import torch.nn as nn
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius



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
    def __init__(self, out_channels=1024, point_feat_dim=256, ratios=[0.5, 0.25]):
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
        expaned_scene_feat = sa3_out[0][point_batch]
        # create edge features by concatenating the scene features with the point features 
        edge_feat = torch.cat([point_feat, expaned_scene_feat], dim=1)
        x, pos, batch = sa3_out

        return x, edge_feat, point_batch

class GraspPredictor(nn.Module):
    def __init__(self, enc_out_channels, predictor_out_size, querry_point_enc_size=16):
        super(GraspPredictor, self).__init__()

        self.predictor = nn.Sequential(
            nn.Linear(enc_out_channels + querry_point_enc_size, enc_out_channels  // 2),
            nn.ReLU(),
            nn.Linear(enc_out_channels // 2, enc_out_channels // 4),
            nn.ReLU(),
            nn.Linear(enc_out_channels // 4, predictor_out_size),
        )

        self.querry_point_encoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, querry_point_enc_size),
        )
    
    def forward(self, edge_features, querry_point, point_batch):
        querry_point_feat = self.querry_point_encoder(querry_point)
        # expanded_querry_point_feat = querry_point_feat[point_batch]
        # h = torch.cat([edge_features, expanded_querry_point_feat], dim=1)
        h = torch.cat([edge_features, querry_point_feat], dim=1)
        # print(f"{h.shape=}")
        h = self.predictor(h)
        return h

class GraspNet(nn.Module):
    def __init__(self, scene_feat_dim, point_feature_dim = 128, predictor_out_size=9):
        super(GraspNet, self).__init__()
        
        self.encoder = Encoder(scene_feat_dim, point_feature_dim)
        self.predictor = GraspPredictor(scene_feat_dim, predictor_out_size)
        # self.evaluator = GraspEvaluator(enc_out_channels, predictor_out_size)
    
    def forward(self, x, pos, batch, querry_point):
        scene_feat, edge_feat, point_batch = self.encoder(x, pos, batch)
        #stack the point features with the scene features
        grasp = self.predictor(scene_feat, querry_point, point_batch)

        trans_m = self.calculateTransformationMatrix(grasp)
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
    model = GraspNet(scene_feat_dim= 1028, point_feature_dim=256, predictor_out_size=9) 

    data = torch.randn((50, 3))
    pos = torch.randn((50, 3))
    batch = torch.arange(50, dtype=torch.long)
    print(pos.shape)
    for i in range(10):
        batch[i*5:(i+1)*5] = i
    query_point = torch.randn((10, 3))
    grasps = model(None, pos, batch, query_point)
    print(grasps.shape)
    # # check the orthogonality of the grasp rotation
    # grasp = grasps[0]
    # print(grasp)
    # orthogonality_check = torch.matmul(grasp.transpose(0, 1), grasp)
    # print(orthogonality_check)
    