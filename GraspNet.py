import torch
import torch.nn as nn
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
        self.sa1_module = SAModule(0.2, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, global_feat_dim]))
    
    def forward(self, x, pos, batch):
        sa0_out = (x, pos, batch)
        sa1_out = self.sa1_module(*sa0_out)
        sa2_out = self.sa2_module(*sa1_out)
        sa3_out = self.sa3_module(*sa2_out)
        return sa0_out, sa1_out, sa2_out, sa3_out
    
# class Encoder(torch.nn.Module):
#     def __init__(self, out_channels=1024, point_feat_dim=256, ratios=[0.5, 0.25]):
#         super().__init__()

#         self.sa1_module = SAModule(ratios[0], 0.2, MLP([3, 64, 64, 128]))
#         self.sa2_module = SAModule(ratios[1], 0.4, MLP([128 + 3, 128, 128, point_feat_dim]))
#         self.sa3_module = GlobalSAModule(MLP([point_feat_dim + 3, point_feat_dim, 512, out_channels]))

#     def forward(self, x, pos, batch):
#         sa1_out = self.sa1_module(x, pos, batch)
#         sa2_out = self.sa2_module(*sa1_out)
#         point_feat, _, point_batch = sa2_out
#         sa3_out = self.sa3_module(*sa2_out)
#         # dublicate the scene features to match the number of points in the batch
#         expaned_scene_feat = sa3_out[0][point_batch]
#         # create edge features by concatenating the scene features with the point features 
#         edge_feat = torch.cat([point_feat, expaned_scene_feat], dim=1)
#         x, pos, batch = sa3_out

#         return x, edge_feat, point_batch


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
    
class GraspPointPredictor(nn.Module):
    def __init__(self, global_feat_dim, approach_feat_dim):
        super(GraspPointPredictor, self).__init__()
        
        self.fp3_module = FPModule(1, MLP([global_feat_dim + 256, 256, 256]))
        self.fp2_module = FPModule(3, MLP([256 + 128, 256, 128]))
        self.fp1_module = FPModule(3, MLP([128 + 3, 128, 128, 128]))

        self.approach_point_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, approach_feat_dim)
        )
        # self.mlp = MLP([128 + approach_feat_dim, 128, 128, 1], dropout=0.5, act=None)
        self.mlp = nn.Sequential(
            nn.Linear(128 + approach_feat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, sa0_out, sa1_out, sa2_out, sa3_out, approach_point):
        fp3_out = self.fp3_module(*sa3_out, *sa2_out)
        fp2_out = self.fp2_module(*fp3_out, *sa1_out)
        x, _, batch_idx = self.fp1_module(*fp2_out, *sa0_out)
        approach_feat = self.approach_point_encoder(approach_point)
        expanded_approach_feat = approach_feat[batch_idx]
        x = torch.cat([x, expanded_approach_feat], dim=1)
        x = self.mlp(x)
        return x
    
class GraspTranslationPredictor(nn.Module):
    def __init__(self, global_feat_dim, approach_feat_dim=64, out_size=3):
        super(GraspTranslationPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(global_feat_dim + approach_feat_dim, global_feat_dim  // 2),
            nn.ReLU(),
            nn.Linear(global_feat_dim // 2, global_feat_dim // 4),
            nn.ReLU(),
            nn.Linear(global_feat_dim // 4, out_size),
        )
        self.approach_and_contact_points_encoder = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, approach_feat_dim)
        )
    
    def forward(self, global_feats, approach_point, grasp_points):
        grasp_points = grasp_points.view(-1, 6)
        conditioned_points = torch.cat([approach_point, grasp_points], dim=1)
        approach_feat = self.approach_and_contact_points_encoder(conditioned_points)
        x = torch.cat([global_feats, approach_feat], dim=1)
        x = self.predictor(x)
        return x

class GraspNet(nn.Module):
    def __init__(self, global_feat_dim, approach_feat_dim = 64, multi_gpu=False, device="cuda"):
        super(GraspNet, self).__init__()
        self.multi_gpu = multi_gpu
        self.device = device
        self.encoder = Encoder(global_feat_dim)
        self.grasp_point_head = GraspPointPredictor(global_feat_dim, approach_feat_dim)
        self.trans_head = GraspTranslationPredictor(global_feat_dim, approach_feat_dim)

        self.mse_loss = nn.MSELoss()
        self.gripper_length = 1.12169998e-01
        self.gripper_length = torch.tensor(self.gripper_length).to(self.device)
    
    def forward(self, data):
        if self.multi_gpu:
            pos, _, batch_idx, approach_point_idx = self.multi_gpu_collate_fn(data)
        else:
            pos, _, batch_idx, approach_point_idx = self.collate_fn(data)
            pos = pos.to(self.device)
            batch_idx = batch_idx.to(self.device)

        sa0_out, sa1_out, sa2_out, global_feat = self.encoder(pos, pos, batch_idx)
        approach_point = pos[approach_point_idx]
        h = self.grasp_point_head(sa0_out, sa1_out, sa2_out, global_feat, approach_point)

        grasp_points = []
        for i in range(len(data)):
            sample_mask = batch_idx == i
            sample_h = h[sample_mask]
            prob_over_sample_points = torch.sigmoid(sample_h)
            #select top 2 points with the highest probability
            prasp_point_indeces = torch.topk(prob_over_sample_points, 2, dim=0).indices
            sample_pos = pos[sample_mask]
            grasp_points.append(sample_pos[prasp_point_indeces])

        grasp_points = torch.stack(grasp_points, dim=0).squeeze()
        global_feat = global_feat[0]
        trans_pos = self.trans_head(global_feat, approach_point, grasp_points)

        grasps, gripper_length, dot_z_x = self.createGrasp(grasp_points, trans_pos)
        return grasps, gripper_length, dot_z_x

    def calculate_loss(self, target, pred, length_pred, dot_z_x):
        # print(pred.shape)
        # print(target.shape)
        grasp_matrix_loss = self.mse_loss(pred, target)
        # print(length_pred.shape)
        # print(self.gripper_length.shape)
        gripper_length_constraint = length_pred - self.gripper_length
        gripper_length_constraint = torch.sum(gripper_length_constraint *  gripper_length_constraint)
        # print(length_pred.shape)
        # print(self.gripper_length.shape)
        dot_z_x_constraint = torch.sum(dot_z_x * dot_z_x)
        return grasp_matrix_loss,  gripper_length_constraint, dot_z_x_constraint

    def multi_gpu_collate_fn(self, data):
        pos = data.pos
        grasps = data.y
        batch_idx = data.batch
        query_point_idx = []
        vertex_count = 0

        for i in range(len(data)):
            query_point_idx.append(data[i].query_point_idx + vertex_count)
            vertex_count += data[i].pos.shape[0]
        query_point_idx = torch.tensor(query_point_idx, dtype=torch.int64)
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
            query_point_idx = sample.query_point_idx
            batch_querry_point_idx.append(query_point_idx + vertex_count)
            vertex_count += points.shape[0]
            batch_idx.extend([i] *  points.shape[0])
            vertices.append(points)
            grasp_gt.append(gt)
                
        batch_idx = torch.tensor(batch_idx, dtype=torch.int64)
        vertices = torch.cat(vertices, dim=0)
        grasp_gt = torch.stack(grasp_gt, dim=0)

        batch_querry_point_idx = torch.tensor(batch_querry_point_idx, dtype=torch.int64)

        return vertices, grasp_gt, batch_idx, batch_querry_point_idx
    
    def createGrasp(self, grasp_points, trans_pos):
        #create the translation vector
        p1 = grasp_points[:, 0]
        p2 = grasp_points[:, 1]
        #create the rotation matrix
        z_axis = p2 - p1
        z_axis = z_axis / torch.norm(z_axis)

        mid_point = (p1 + p2) / 2
        x_axis = trans_pos - mid_point
        x_axis = x_axis / torch.norm(x_axis)
        y_axis = torch.cross(z_axis, x_axis)
        y_axis = y_axis / torch.norm(y_axis)

        grasp = torch.eye(4).to(grasp_points.device)
        grasps = grasp.repeat(grasp_points.shape[0], 1, 1)
        r = torch.stack([x_axis, y_axis, z_axis], dim=1)
        grasps[:, :3, :3] = r
        grasps[:, :3, 3] = trans_pos
        grasps = grasps.view(-1, 16)


        gripper_length = torch.norm(trans_pos - mid_point) 
        dot_z_x = z_axis * x_axis
        dot_z_x = torch.sum(dot_z_x, dim=1)

        return grasps, gripper_length, dot_z_x

if __name__ == "__main__":

    from acronym_dataset import AcronymDataset
    from create_dataset_paths import save_split_meshes
    from torch_geometric.loader import DataListLoader
    from acronym_dataset import RandomRotationTransform

    model = GraspNet(global_feat_dim=1024, device="cpu") 
    train_paths, val_paths = save_split_meshes('../data', 100)
    rotation_range = [-180, 180]

    # transform = RandomRotationTransform(rotation_range)
    train_dataset = AcronymDataset(train_paths, crop_radius=None, transform=None, normalize_vertices=True)
    train_loader = DataListLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)

    for i, data in enumerate(train_loader):
        pred, gripper_length, dot_z_x = model(data)
        grasp_gt = torch.stack([sample.y for sample in data], dim=0)

        loss = model.calculate_loss(grasp_gt, pred, gripper_length, dot_z_x)
        print(pred.shape)
        print(gripper_length)
        print(dot_z_x)
        break
    # # check the orthogonality of the grasp rotation
    # grasp = grasps[0]
    # print(grasp)
    # orthogonality_check = torch.matmul(grasp.transpose(0, 1), grasp)
    # print(orthogonality_check)
    