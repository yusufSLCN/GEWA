import torch
import torch.nn as nn
from torch_geometric.nn import MLP
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
    def forward(self, edge_features):
        # expanded_querry_point_feat = querry_point_feat[point_batch]
        # h = torch.cat([edge_features, expanded_querry_point_feat], dim=1)
        h = self.predictor(edge_features)
        return h

class Encoder(torch.nn.Module):
    def __init__(self, out_channels, point_feat_dim=256, ratios=[1.0, 1.0]):
        super().__init__()

        self.sa1_module = SAModule(ratios[0], 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(ratios[1], 0.4, MLP([128 + 3, 128, 128, point_feat_dim]))
        self.sa3_module = GlobalSAModule(MLP([point_feat_dim + 3, point_feat_dim, 512, out_channels]))

    def forward(self, x, pos, batch, query_point_idx):
        sa1_out = self.sa1_module(x, pos, batch)
        sa2_out = self.sa2_module(*sa1_out)
        point_feat, _, point_batch = sa2_out
        sa3_out = self.sa3_module(*sa2_out)
        # expanded_scene_feat = sa3_out[0][point_batch]
        scene_feat = sa3_out[0]
        point_feat = point_feat[query_point_idx]

        # create edge features by concatenating the scene features with the point features 
        edge_feat = torch.cat([point_feat, scene_feat], dim=1)
        x, pos, batch = sa3_out

        return x, edge_feat, point_batch
    
class GewaNet(nn.Module):
    def __init__(self, scene_feat_dim=1024, point_feat_dim=256, predictor_out_size = 9, multi_gpu=False):
        super(GewaNet, self).__init__()
        self.encoder = Encoder(out_channels = scene_feat_dim, point_feat_dim=point_feat_dim)
        self.predictor = EdgeGraspPredictor(scene_feat_dim, point_feat_dim=point_feat_dim, out_size=predictor_out_size)
        self.multi_gpu = multi_gpu
    def forward(self, data):
        if self.multi_gpu:
            pos, grasps, batch_idx, query_point_idx = self.multi_gpu_collate_fn(data)
        else:
            pos, grasps, batch_idx, query_point_idx = self.collate_fn(data)
            
        x, edge_feat, point_batch = self.encoder(None, pos, batch_idx, query_point_idx)
        output = self.predictor(edge_feat)
        trans_m = self.calculateTransformationMatrix(output)
        grasp = trans_m.view(-1, 16)
        return grasp
    
    def multi_gpu_collate_fn(self, data):
        pos = data.pos
        grasps = data.y
        batch_idx = data.batch
        query_point_idx = data.sample_info["query_point_idx"]
        return pos, grasps, batch_idx, query_point_idx

    def collate_fn(self, batch):
        batch_idx = []
        vertices = []
        grasp_gt = []
        batch_querry_point_idx = []
        vertex_count = 0
        for i, sample in enumerate(batch):
            points = sample.pos
            gt = sample.y
            info = sample.sample_info
            batch_querry_point_idx.append(info['query_point_idx'] + vertex_count)
            vertex_count = points.shape[0]
            batch_idx.extend([i] * vertex_count)
            vertices.append(points)
            grasp_gt.append(gt)
                
        batch_idx = torch.tensor(batch_idx, dtype=torch.int64)
        vertices = torch.cat(vertices, dim=0)
        grasp_gt = torch.stack(grasp_gt, dim=0)
        batch_querry_point_idx = torch.tensor(batch_querry_point_idx, dtype=torch.int64)

        return vertices, grasp_gt, batch_idx, batch_querry_point_idx
    
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
    model = GewaNet(scene_feat_dim= 1024, point_feat_dim=256, predictor_out_size=9) 

    num_meshes = 2
    mesh_vertex_count = 50
    total_vertex_count = num_meshes * mesh_vertex_count
    data = torch.randn((total_vertex_count, 3))
    pos = torch.randn((total_vertex_count, 3))
    batch = torch.arange(total_vertex_count, dtype=torch.long)
    query_point_idx = torch.zeros(num_meshes, dtype=torch.long)
    for i in range(num_meshes):
        batch[i*mesh_vertex_count:(i+1)*mesh_vertex_count] = i
        query_point_idx[i] = i * mesh_vertex_count

    grasps = model(None, pos, batch, query_point_idx)