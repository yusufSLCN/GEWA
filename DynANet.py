import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool

class DynANet(nn.Module):
    def __init__(self, grasp_dim=16, k=16, num_grasp_sample=500):
        super(DynANet, self).__init__()
        
        self.num_grasp_sample = num_grasp_sample
        self.grap_dim = grasp_dim
        self.k = k
        self.multi_gpu = False
        
        self.conv1 = DynamicEdgeConv(MLP([6, 16, 16, 32]), k=self.k, aggr='max')
        self.conv2 = DynamicEdgeConv(MLP([64, 64, 64, 128]), k=self.k, aggr='max')
        self.conv3 = DynamicEdgeConv(MLP([256, 256, 256, 512]), k=self.k, aggr='max')
        
        self.shared_mlp = MLP([512 + 128 + 32, 256, 128])
        
        # Classification head (per-point)
        self.classification_head = nn.Linear(128, 1)
        
        # Grasp prediction head
        self.grasp_head = nn.Sequential(
            nn.Linear(512 + 128 + 32 + 128 + 3, 64),  # 512 + 128 + 32  local features,  128 from global embedding, 3 from selected point
            nn.ReLU(),
            nn.Linear(64, self.grap_dim)
        )

    def forward(self, data, temperature=1.0):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        
        x = torch.cat([x1, x2, x3], dim=-1)
        
        shared_features = self.shared_mlp(x)
        
        # Classification output (per-point)
        classification_logits = self.classification_head(shared_features).squeeze(-1)
        classification_output = torch.sigmoid(classification_logits)
        
        # Global point cloud embedding
        global_embedding = global_max_pool(shared_features, batch)
        
        # Point selection (per point cloud)
        # scaled_logits = classification_logits / temperature
        
        point_distributions = []
        approach_points = []
        grasp_gt = []
        approach_point_idxs = []
        
        for i in range(batch.max() + 1):  # Iterate over each point cloud in the batch
            mask = (batch == i)
            sample_pos = pos[mask]
            sample_grasps = data.y[mask]
            
            # Multinomial sampling for point selection
            if self.training:
                approach_scores = data.approach_scores[mask]
                with_replacement = torch.sum(approach_scores > 0.5) < self.num_grasp_sample
                point_index = torch.multinomial(approach_scores, num_samples=self.num_grasp_sample,
                                                replacement=with_replacement.item())
            else:
                sample_appraoch_prob = classification_output[mask]
                with_replacement = torch.sum(sample_appraoch_prob > 0.5) < self.num_grasp_sample
                point_index = torch.multinomial(sample_appraoch_prob, num_samples=self.num_grasp_sample, 
                                                replacement=with_replacement.item())
 

            # point_index = torch.multinomial(da, num_samples=self.num_grasp_sample)
            # point_index = torch.arange(len(sample_pos))
            selected_point = sample_pos[point_index].squeeze(0)
            approach_points.append(selected_point)
            #FOR TEST
            # dummy_gt = torch.eye(4).reshape(-1).repeat(len(point_index), 1).to(sample_grasps.device)
            # grasp_gt.append(dummy_gt)

            grasp_gt.append(sample_grasps[point_index].reshape(-1, 16))
            approach_point_idxs.append(point_index + i * sample_pos.shape[0])
        
        approach_points = torch.cat(approach_points)
        grasp_gt = torch.cat(grasp_gt)
        approach_point_idxs = torch.cat(approach_point_idxs)
        # Grasp prediction for the entire batch
        repeated_global_embedding = global_embedding.repeat(self.num_grasp_sample, 1)
        selected_local_features = x[approach_point_idxs]
        grasp_features = torch.cat([selected_local_features, repeated_global_embedding, approach_points], dim=1)

        grasp_outputs = self.grasp_head(grasp_features)
        if self.grap_dim != 16:
            grasp_outputs = self.calculateTransformationMatrix(grasp_outputs, approach_points)
            grasp_outputs = grasp_outputs.view(-1, 16)
        else:
            grasp_outputs = grasp_outputs.reshape(-1, 4, 4)
            grasp_outputs[:, :3, 3] + approach_points
            grasp_outputs = grasp_outputs.view(-1, 16)
        
        return classification_output, grasp_outputs, approach_points, grasp_gt
    
    def calculateTransformationMatrix(self, grasp, approach_points):
        translation = grasp[:, :3] + approach_points
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

    model = DynANet() 
    train_paths, val_paths = save_split_samples('../data', -1)
    classification_criterion = nn.BCELoss()
    grasp_criterion = nn.MSELoss()
    # transform = RandomRotationTransform(rotation_range)
    train_dataset = GewaDataset(train_paths, transform=None, normalize_points=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    num_success= 0
    for i, data in enumerate(train_loader):
        print(f"Batch: {i}/{len(train_loader)}")
        classification_output, grasp_outputs, approach_points, grasp_gt = model(data)
        approach_score_gt = data.approach_scores
        approach_score_gt = (approach_score_gt > 0).float()
        # print(approach_score_gt.shape, grasp_target.shape)
        classification_loss = classification_criterion(classification_output, approach_score_gt)
        grasp_loss = grasp_criterion(grasp_outputs, grasp_gt)
        grasp_outputs = grasp_outputs.reshape(-1, 4, 4).detach().numpy()
        grasp_gt = grasp_gt.reshape(-1, 4, 4).detach().numpy()
        num_success += check_batch_grasp_success(grasp_outputs, grasp_gt, 0.03, np.deg2rad(30))
        # grasp_gt = torch.stack([sample.y for sample in data], dim=0)
        # approach_gt = torch.stack([sample.approach_point_idx for sample in data], dim=0)
        # loss = model.calculate_loss(grasp_gt, grasp_pred, approach_gt, approch_pred)
    
    print(f"Success rate: {num_success / len(train_dataset)}")