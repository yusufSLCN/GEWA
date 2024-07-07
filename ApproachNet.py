import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate
import numpy as np
from define_new_axis import define_new_axis

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=True)

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
    
class Encoder(nn.Module):
    def __init__(self, global_feat_dim):
        super(Encoder, self).__init__()
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, global_feat_dim]))
    
    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        return sa0_out, sa1_out, sa2_out, sa3_out


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
    
class ApproachPointPredictor(nn.Module):
    def __init__(self, global_feat_dim):
        super(ApproachPointPredictor, self).__init__()
        
        self.fp3_module = FPModule(1, MLP([global_feat_dim + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        # self.mlp = MLP([128 + approach_feat_dim, 128, 128, 1], dropout=0.5, act=None)
        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, sa0_out, sa1_out, sa2_out, sa3_out):
        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        x = self.mlp(x)
        return x
    
class GraspPosePredictor(nn.Module):
    def __init__(self, global_feat_dim, approach_feat_dim=64, grasp_dim=16):
        super(GraspPosePredictor, self).__init__()
        
        self.grasp_predictor = nn.Sequential(
            nn.Linear(global_feat_dim + approach_feat_dim, global_feat_dim  // 2),
            nn.ReLU(),
            nn.Linear(global_feat_dim // 2, global_feat_dim // 4),
            nn.ReLU(),
            nn.Linear(global_feat_dim // 4, grasp_dim),
        )
        self.approach_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, approach_feat_dim)
        )
    
    def forward(self, global_feats, approach_point):
        approach_feat = self.approach_encoder(approach_point)
        x = torch.cat([global_feats, approach_feat], dim=1)
        grasp = self.grasp_predictor(x)
        return grasp

class ApproachNet(nn.Module):
    def __init__(self, global_feat_dim, approach_feat_dim=64, multi_gpu=False, device="cuda"):
        super(ApproachNet, self).__init__()
        self.multi_gpu = multi_gpu
        self.device = device
        self.encoder = Encoder(global_feat_dim)
        self.approach_head = ApproachPointPredictor(global_feat_dim)
        self.grasp_head = GraspPosePredictor(global_feat_dim, approach_feat_dim)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, data):
        if self.multi_gpu:
            pos, point_grasp_list, batch_idx, approach_gt = self.multi_gpu_collate_fn(data)
        else:
            pos, point_grasp_list, batch_idx, approach_gt = self.collate_fn(data)
            pos = pos.to(self.device)
            batch_idx = batch_idx.to(self.device)

        sa0_out, sa1_out, sa2_out, global_out = self.encoder(pos, pos, batch_idx)
        # approach_point = pos[approach_point_idx]
        approach_out = self.approach_head(sa0_out, sa1_out, sa2_out, global_out)

        approach_points = []
        approach_point_idxs = []
        approach_dist = []
        grasp_gt = []
        # print(point_grasp_list.shape)
        for i in range(batch_idx[-1] + 1):
            batch_mask = batch_idx == i
            pos_i = pos[batch_mask]
            approach_batch = approach_out[batch_mask]
            dist = F.log_softmax(approach_batch, dim=0)
            approach_dist.append(dist.squeeze())
            approach_point_idx = torch.argmax(approach_batch, dim=0)
            grasp_gt.append(point_grasp_list[i][approach_point_idx].squeeze())
            approach_point = pos_i[approach_point_idx]
            approach_points.append(approach_point)
            approach_point_idxs.append(approach_point_idx)

        approach_points = torch.stack(approach_points, dim=0).squeeze()
        approach_point_idxs = torch.stack(approach_point_idxs, dim=0)
        grasp_gt = torch.stack(grasp_gt, dim=0)
        # approach_dist = torch.stack(approach_dist, dim=0)
        global_feat = global_out[0]
        grasp_pred = self.grasp_head(global_feat, approach_points)

        grasp_loss, approach_loss = self.calculate_loss(grasp_gt, grasp_pred, approach_gt, approach_dist)
        return grasp_pred, approach_dist, grasp_loss, approach_loss
    
    def calculate_loss(self, grasp_gt, grasp_pred, approach_gt, approach_dist):
        grasp_loss = self.mse_loss(grasp_pred, grasp_gt)
        approch_loss = 0
        for a_dist, a_gt in zip(approach_dist, approach_gt):
            # print(a_dist.shape, a_gt.shape)
            approch_loss += self.bce_loss(F.sigmoid(a_dist), a_gt)
        approch_loss = approch_loss / len(approach_gt)
        return grasp_loss, approch_loss

    def multi_gpu_collate_fn(self, data):
        pos = data.pos
        point_grasp_list = data.point_grasp_list
        batch_idx = data.batch
        approch_dist = []

        for i in range(len(data)):
            approch_dist.append(F.log_softmax(data[i].approach))
        
        approch_dist = torch.stack(approch_dist, dim=0)
        return pos, point_grasp_list, batch_idx, approch_dist

    def collate_fn(self, batch):
        batch_idx = []
        vertices = []
        batch_querry_point_idx = []
        vertex_count = 0
        approch_gt = []
        point_grasp_list = []

        for i, sample in enumerate(batch):
            points = sample.pos
            query_point_idx = sample.query_point_idx
            batch_querry_point_idx.append(query_point_idx + vertex_count)
            vertex_count += points.shape[0]
            batch_idx.extend([i] *  points.shape[0])
            vertices.append(points)
            point_grasp_list.append(sample.point_grasp_list.reshape(-1, 16))
            approach = torch.tensor(sample.approach > 0, dtype=torch.float32)
            approch_gt.append(approach)

                
        batch_idx = torch.tensor(batch_idx, dtype=torch.int64)
        vertices = torch.cat(vertices, dim=0)
        # point_grasp_list = torch.cat(point_grasp_list, dim=0)
        # approch_dist = torch.cat(approch_dist, dim=0)


        return vertices, point_grasp_list, batch_idx, approch_gt
    

    def createGrasp(self, grasp_points, rotations):
        #create the translation vector
        p1 = grasp_points[:, 0]
        p2 = grasp_points[:, 1]
        #create the rotation matrix
        z_axis = p2 - p1
        z_axis = z_axis / torch.norm(z_axis)
        mid_point, x_axis = [], []
        for i in range(len(p1)):
            mid_point_i, x_axis_i = define_new_axis(p1[i], p2[i], rotations[i])
            mid_point.append(mid_point_i)
            x_axis.append(x_axis_i)
        mid_point = torch.stack(mid_point, dim=0)
        x_axis = torch.stack(x_axis, dim=0)
        translation = mid_point + x_axis * self.gripper_length

        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)

        grasp = torch.eye(4).to(grasp_points.device)
        grasps = grasp.repeat(grasp_points.shape[0], 1, 1)
        r = torch.stack([x_axis, y_axis, z_axis], dim=1)
        grasps[:, :3, :3] = r
        grasps[:, :3, 3] = translation
        grasps = grasps.view(-1, 16)


        gripper_length = torch.norm(translation - mid_point) 
        dot_z_x = z_axis * x_axis
        dot_z_x = torch.sum(dot_z_x, dim=1)

        return grasps, gripper_length, dot_z_x

if __name__ == "__main__":

    from acronym_dataset import AcronymDataset
    from create_dataset_paths import save_split_meshes
    from torch_geometric.loader import DataListLoader
    from acronym_dataset import RandomRotationTransform

    model = ApproachNet(global_feat_dim=1024, device="cpu") 
    train_paths, val_paths = save_split_meshes('../data', 100)
    rotation_range = [-180, 180]

    # transform = RandomRotationTransform(rotation_range)
    train_dataset = AcronymDataset(train_paths, crop_radius=None, transform=None, normalize_vertices=True)
    train_loader = DataListLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)

    for i, data in enumerate(train_loader):
        grasp_pred, approch_pred = model(data)
        # grasp_gt = torch.stack([sample.y for sample in data], dim=0)
        # approach_gt = torch.stack([sample.approach_point_idx for sample in data], dim=0)
        # loss = model.calculate_loss(grasp_gt, grasp_pred, approach_gt, approch_pred)
        break
    # # check the orthogonality of the grasp rotation
    # grasp = grasps[0]
    # print(grasp)
    # orthogonality_check = torch.matmul(grasp.transpose(0, 1), grasp)
    # print(orthogonality_check)
    