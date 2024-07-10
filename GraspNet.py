import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool

class GraspNet(nn.Module):
    def __init__(self, k=20):
        super(GraspNet, self).__init__()
        
        self.k = k
        
        self.conv1 = DynamicEdgeConv(MLP([6, 64, 64]), k=self.k, aggr='max')
        self.conv2 = DynamicEdgeConv(MLP([128, 128]), k=self.k, aggr='max')
        self.conv3 = DynamicEdgeConv(MLP([256, 256]), k=self.k, aggr='max')
        
        self.shared_mlp = MLP([256 + 128 + 64, 256, 128])
        
        # Binary classification head (per-point)
        self.approach_score_head = nn.Linear(128, 1)
        
        # 16-dimensional vector prediction head (global)
        self.grasp_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )

    def forward(self, data):
        x, batch = data.pos, data.batch
        
        x1 = self.conv1(x, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        
        x = torch.cat([x1, x2, x3], dim=-1)
        
        shared_features = self.shared_mlp(x)
        
        # Binary classification output (per-point)
        approach_score_pred = torch.sigmoid(self.approach_score_head(shared_features)).squeeze(-1)
        
        # 16-dimensional vector output (global)
        # pooled_features = global_max_pool(shared_features, batch)
        grasp_pred = self.grasp_head(shared_features)
        
        return approach_score_pred, grasp_pred

if __name__ == "__main__":

    from gewa_dataset import GewaDataset
    from create_gewa_dataset import save_split_samples
    from torch_geometric.loader import DataLoader

    model = GraspNet() 
    train_paths, val_paths = save_split_samples('../data', 100)

    # transform = RandomRotationTransform(rotation_range)
    train_dataset = GewaDataset(train_paths, transform=None, normalize_points=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)

    def combined_loss(approach_score_pred, grasp_pred, approach_score_gt, grasp_target):
        classification_loss = F.binary_cross_entropy(approach_score_pred, approach_score_gt)
        vector_loss = F.mse_loss(grasp_pred, grasp_target)
        return classification_loss + vector_loss
    
    for i, data in enumerate(train_loader):
        print(data)
        approach_score_pred, grasp_pred = model(data)

        approach_score_gt = data.approach_scores
        approach_score_gt = (approach_score_gt > 0.5).float()
        grasp_target = data.y.reshape(-1, 16)
        print(approach_score_gt.shape)
        loss = combined_loss(approach_score_pred, grasp_pred, approach_score_gt, grasp_target)
        # grasp_gt = torch.stack([sample.y for sample in data], dim=0)

        # loss = model.calculate_loss(grasp_gt, pred, gripper_length, dot_z_x)
        # print(pred.shape)
        # print(gripper_length)
        # print(dot_z_x)
        break
    # # check the orthogonality of the grasp rotation
    # grasp = grasps[0]
    # print(grasp)
    # orthogonality_check = torch.matmul(grasp.transpose(0, 1), grasp)
    # print(orthogonality_check)
    