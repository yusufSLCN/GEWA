import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn_interpolate

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
        # self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa1_module = SAModule(1., 0.1, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(1., 0.2, MLP([128 + 3, 128, 128, 256]))

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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sa0_out, sa1_out, sa2_out, sa3_out):
        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, _ = self.fp1_module(*fp2_out, *sa0_out)
        x = self.mlp(x)
        return x
    
# class ApproachPointClassifier(nn.Module):

    
class GraspPosePredictor(nn.Module):
    def __init__(self, global_feat_dim, approach_feat_dim=64, grasp_dim=16):
        super(GraspPosePredictor, self).__init__()
        
        self.approach_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, approach_feat_dim)
        )

        self.grasp_predictor = nn.Sequential(
            nn.Linear(global_feat_dim + approach_feat_dim + 128 + 256, global_feat_dim  // 2),
            nn.ReLU(),
            nn.Linear(global_feat_dim // 2, global_feat_dim // 4),
            nn.ReLU(),
            nn.Linear(global_feat_dim // 4, grasp_dim),
        )

    
    def forward(self, global_feats, approach_point):
        approach_feat = self.approach_encoder(approach_point)
        if approach_feat.dim() == 1:
            approach_feat = approach_feat.unsqueeze(0)
        x = torch.cat([global_feats, approach_feat], dim=1)
        grasp = self.grasp_predictor(x)
        return grasp

class ApproachNet(nn.Module):
    def __init__(self, global_feat_dim, approach_feat_dim=64, grasp_dim=16, num_grasp_sample=1000, multi_gpu=False, device="cuda"):
        super(ApproachNet, self).__init__()
        self.multi_gpu = multi_gpu
        self.device = device
        self.grasp_dim = grasp_dim
        self.num_grasp_sample = num_grasp_sample
        self.encoder = Encoder(global_feat_dim)
        self.approach_head = ApproachPointPredictor(global_feat_dim)
        self.grasp_head = GraspPosePredictor(global_feat_dim, approach_feat_dim, grasp_dim)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, data):
        pos = data.pos
        point_grasp_list = data.y
        batch_idx = data.batch

        sa0_out, sa1_out, sa2_out, global_out = self.encoder(pos, pos, batch_idx)

        # print(sa0_out[0].shape, sa1_out[0].shape, sa2_out[0].shape, global_out[0].shape)
        # approach_point = pos[approach_point_idx]
        approach_out = self.approach_head(sa0_out, sa1_out, sa2_out, global_out)

        approach_points = []
        approach_point_idxs = []
        approach_score_pred = []
        grasp_gt = []
        # print(point_grasp_list.shape)
        for i in range(batch_idx[-1] + 1):
            batch_mask = batch_idx == i
            pos_i = pos[batch_mask]
            approach_i = approach_out[batch_mask].squeeze()
            point_grasps_i = point_grasp_list[batch_mask]
            # dist = F.log_softmax(approach_batch, dim=0)
            approach_score_pred.append(approach_i)

            # approach_point_idx = torch.argmax(approach_batch, dim=0)
            approach_point_idx = torch.multinomial(approach_i, self.num_grasp_sample)
            grasp = point_grasps_i[approach_point_idx].squeeze()
            grasp_gt.append(grasp)
            approach_point = pos_i[approach_point_idx]
            approach_points.append(approach_point)
            approach_point_idxs.append(approach_point_idx + i * len(pos_i))

        #prepare input for grasp prediction
        approach_points = torch.stack(approach_points, dim=0).reshape(-1, 3)
        approach_point_idxs = torch.cat(approach_point_idxs, dim=0)
        local_features = torch.cat((sa1_out[0], sa2_out[0]), dim=1)
        #select the grasp features for the selected approach points
        selected_grasp_features = local_features[approach_point_idxs]
        repeated_global_out = global_out[0].repeat(self.num_grasp_sample, 1) #repeat the global feature for each point cloud point
        selected_grasp_features = torch.cat([selected_grasp_features, repeated_global_out], dim=1)
        
        grasp_pred = self.grasp_head(selected_grasp_features, approach_points)
        if self.grasp_dim != 16:
            grasp_pred = self.calculateTransformationMatrix(grasp_pred, approach_points)
            grasp_pred = grasp_pred.view(-1, 16)

        approach_score_pred = torch.stack(approach_score_pred, dim=0)
        grasp_gt = torch.stack(grasp_gt, dim=0).reshape(-1, 16)

        grasp_loss, approach_loss = self.calculate_loss(grasp_gt, grasp_pred, data.approach_scores, approach_score_pred)
        return grasp_pred, approach_score_pred, grasp_gt, grasp_loss, approach_loss, approach_points
    
    def calculate_loss(self, grasp_gt, grasp_pred, approach_scores, approach_pred):
        approach_gt = approach_scores > 0
        approach_gt = approach_gt.float().reshape(-1, approach_pred.shape[-1])
        # print(grasp_gt.shape, grasp_pred.shape)
        grasp_loss = self.mse_loss(grasp_pred, grasp_gt)
        approch_loss = 0
        for a_dist, a_gt in zip(approach_pred, approach_gt):
            approch_loss += self.bce_loss(a_dist, a_gt)
        approch_loss = approch_loss / len(approach_gt)
        return grasp_loss, approch_loss
    

    def calculateTransformationMatrix(self, grasp, points):
        translation = grasp[:, :3] + points
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

    from gewa_dataset import GewaDataset
    from create_gewa_dataset import save_split_samples
    from torch_geometric.loader import DataLoader
    from metrics import check_batch_grasp_success
    import numpy as np

    model = ApproachNet(global_feat_dim=1024, num_grasp_sample=500, device="cpu") 
    train_paths, val_paths = save_split_samples('../data', -1)

    # transform = RandomRotationTransform(rotation_range)
    train_dataset = GewaDataset(train_paths, transform=None, normalize_points=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    num_success= 0
    for i, data in enumerate(train_loader):
        print(f"Batch: {i}/{len(train_loader)}")
        grasp_pred, approach_score_pred, grasp_gt, grasp_loss, approach_loss, approach_points = model(data)

        grasp_pred = grasp_pred.reshape(-1, 4, 4).detach().numpy()
        grasp_gt = grasp_gt.reshape(-1, 4, 4).detach().numpy()
        num_success += check_batch_grasp_success(grasp_gt, grasp_gt, 0.03, np.deg2rad(30))
        # grasp_gt = torch.stack([sample.y for sample in data], dim=0)
        # approach_gt = torch.stack([sample.approach_point_idx for sample in data], dim=0)
        # loss = model.calculate_loss(grasp_gt, grasp_pred, approach_gt, approch_pred)
    
    print(f"Success rate: {num_success / len(train_dataset)}")
    