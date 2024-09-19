import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, global_mean_pool
import numpy as np

class TppAngleNet(nn.Module):
    def __init__(self, k=8, angle_bin_count=72, num_grasp_sample=100, num_points=1000, max_num_grasps=10, only_classifier=False):
        super(TppAngleNet, self).__init__()
        
        self.num_grasp_sample = num_grasp_sample
        self.k = k
        self.multi_gpu = False
        self.num_points = num_points
        self.max_num_grasps = max_num_grasps
        self.only_classifier = only_classifier
        self.triu = torch.triu_indices(num_points, num_points, offset=1)

        self.unit_vector = torch.tensor([0, 0, 1], dtype=torch.float32)
        self.num_pairs = self.triu.shape[1]
        
        self.conv1 = DynamicEdgeConv(MLP([6, 16, 32]), k=self.k, aggr='max')
        self.conv2 = DynamicEdgeConv(MLP([64, 64, 128]), k=self.k, aggr='max')
        self.conv3 = DynamicEdgeConv(MLP([256, 256, 512]), k=self.k, aggr='max')
        # self.conv4 = DynamicEdgeConv(MLP([1024, 1024, 2048]), k=self.k, aggr='max')
        # self.conv5 = DynamicEdgeConv(MLP([256, 256, 512]), k=self.k, aggr='max')
        
        self.point_feat_dim = 128
        self.shared_mlp = nn.Sequential(
            MLP([512 + 128 + 32, 256, self.point_feat_dim]),
            # nn.Linear(128, self.point_feat_dim)
        ) 

        self.angle_delta = 2 * np.pi / angle_bin_count
        self.edge_classifier = nn.Sequential(
            nn.Linear(self.point_feat_dim * 3, self.point_feat_dim),
            nn.ReLU(),
            nn.Linear(self.point_feat_dim, 1)
            
        )
        
        # Grasp angle prediction head
        self.angle_head = nn.Sequential(
            nn.Linear(self.point_feat_dim * 3, self.point_feat_dim),
            nn.ReLU(),
            nn.Linear(self.point_feat_dim, self.point_feat_dim),
            nn.ReLU(),
            nn.Linear(self.point_feat_dim, angle_bin_count)
        )

        # self.translation_head = nn.Sequential(
        #     nn.Linear(self.point_feat_dim * 3, self.point_feat_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.point_feat_dim, self.point_feat_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.point_feat_dim, 3),
        # )

    def forward(self, data):
        pos, batch = data.pos, data.batch
        x1 = self.conv1(pos, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        # x4 = self.conv4(x3, batch)
        
        x = torch.cat([x1, x2, x3], dim=-1)
        
        shared_features = self.shared_mlp(x)

        global_embedding = global_max_pool(shared_features, batch)
        shared_features = shared_features.reshape(-1, self.num_points, self.point_feat_dim)
        feat_i = shared_features[:, self.triu[0], :]
        feat_j = shared_features[:, self.triu[1], :]
        global_embedding = global_embedding.reshape(-1, 1, self.point_feat_dim)
        global_embedding = global_embedding.repeat(1, feat_j.shape[1], 1)

        edge_feature_ij = torch.cat([feat_i, feat_j, global_embedding], dim=-1)
        mlp_out_ij = self.edge_classifier(edge_feature_ij)
        mlp_out_ij = mlp_out_ij.squeeze(-1)
        pair_classification_out_ij = torch.sigmoid(mlp_out_ij)

        if self.only_classifier:
            return None, None, None, None, None, pair_classification_out_ij, mlp_out_ij

        
        selected_edge_idxs = []
        selected_edge_features = []
        grasp_gt = np.zeros((self.num_grasp_sample, self.max_num_grasps, 4, 4))
        mid_edge_pos = []
        grasp_touch_points = []
        grasp_axises = []

        edge_scores = data.pair_scores
        edge_scores = edge_scores.reshape(-1, self.num_pairs)
        pos = pos.reshape(-1, self.num_points, 3)
        grasp_gt = np.zeros((pos.shape[0], self.num_grasp_sample, self.max_num_grasps, 4, 4))
        num_valid_grasps = np.zeros((pos.shape[0], self.num_grasp_sample))

        for i in range(edge_scores.shape[0]):  # Iterate over each point cloud in the batch
            sample_pos = pos[i]
            
            # Multinomial sampling for point selection
            if self.training:
                sample_edge_scores = edge_scores[i]
                sample_edge_prob = (sample_edge_scores > 0.5).float()
                with_replacement = torch.sum(sample_edge_prob) < self.num_grasp_sample
                edge_index = torch.multinomial(sample_edge_prob, num_samples=self.num_grasp_sample,
                                                replacement=with_replacement.item())
                # true_idxs = (sample_edge_scores > 0.5).nonzero()

                # edge_index = true_idxs[:self.num_grasp_sample]
            else:
                sample_edge_prob = pair_classification_out_ij[i]
                pos_pair_count = torch.sum(sample_edge_prob > 0.5)
                if pos_pair_count > 0:
                    sample_edge_prob[sample_edge_prob < 0.5] = 0
                with_replacement = pos_pair_count < self.num_grasp_sample
                edge_index = torch.multinomial(sample_edge_prob, num_samples=self.num_grasp_sample, 
                                                replacement=with_replacement.item())
            
            # print("find edge indxs")
            edge_index = edge_index.to("cpu")
            selected_edge_node1 = self.triu[0][edge_index]
            selected_edge_node2 = self.triu[1][edge_index]
            selected_edge_idxs.append(edge_index)
            # print(data.y)
            # print("selection loop start")
            for edge_j, (point1_idx, point2_idx) in enumerate(zip(selected_edge_node1, selected_edge_node2)):
                point1_idx = point1_idx.item()
                point2_idx = point2_idx.item()
                grasp_key = frozenset((point1_idx, point2_idx))
                sample_grasp_dict = data.y[i][0]
                if grasp_key in sample_grasp_dict:
                    edge_grasps = sample_grasp_dict[grasp_key]

                    if len(edge_grasps) > self.max_num_grasps:
                        edge_grasps = edge_grasps[:self.max_num_grasps]

                    if len(edge_grasps) > 0:
                        # edge_grasps = torch.eye(4).reshape(1, 4, 4).repeat(self.max_num_grasps, 1, 1)
                        # grasp_gt[i, edge_j] = edge_grasps
                        grasp_gt[i, edge_j, :len(edge_grasps)] = edge_grasps
                        num_valid_grasps[i, edge_j] = len(edge_grasps)

                # sample_edge_grasps.append(edge_grasps)
                grasp_axis = sample_pos[point2_idx] - sample_pos[point1_idx]
                grasp_axises.append(grasp_axis)
                
                mid_edge_pos.append((sample_pos[point1_idx] + sample_pos[point2_idx]) / 2)
                touch_point_pair = torch.hstack([sample_pos[point1_idx], sample_pos[point2_idx]])
                grasp_touch_points.append(touch_point_pair)

            selected_edge_features.append(edge_feature_ij[i, edge_index])

        # prepare grasp features
        selected_edge_features = torch.stack(selected_edge_features)
        # grasp_touch_points = torch.stack(grasp_touch_points)
        # grasp_touch_points = grasp_touch_points.reshape(-1, self.num_grasp_sample, 6)
        # grasp_features = torch.cat([selected_edge_features, grasp_touch_points], dim=-1)
        
        # predict grasps
        angle_logit = self.angle_head(selected_edge_features)
        angle_prob = F.softmax(angle_logit, dim=-1)
        grasp_angles = torch.argmax(angle_prob, dim=-1)
        grasp_angles = grasp_angles * self.angle_delta
        grasp_angles = grasp_angles.reshape(-1, 1)

        grasp_axises = torch.stack(grasp_axises)
        mid_edge_pos = torch.stack(mid_edge_pos)
        selected_edge_idxs = torch.stack(selected_edge_idxs)
        grasp_gt = torch.tensor(grasp_gt, dtype=torch.float32)
        num_valid_grasps = torch.tensor(num_valid_grasps, dtype=torch.int8)
        # print(grasp_angles.shape, grasp_axises.shape, mid_edge_pos.shape)

        
        grasp_pred = self.calculateTransformationMatrix(grasp_axises, mid_edge_pos, grasp_angles)

        # print("return")
        selected_edge_idxs = selected_edge_idxs.to(grasp_angles.device)
        grasp_gt = grasp_gt.to(grasp_angles.device)
        num_valid_grasps = num_valid_grasps.to(grasp_angles.device)
        grasp_pred = grasp_pred.to(grasp_angles.device)
        # print(grasp_outputs.device, selected_edge_idxs.device,
        #        mid_edge_pos.device, grasp_gt.device, num_valid_grasps.device, pair_classification_out_ij.device, mlp_out_ij.device)


        return grasp_pred, selected_edge_idxs, mid_edge_pos, grasp_gt, num_valid_grasps, pair_classification_out_ij, mlp_out_ij

    def rotate_vector(self, v, k, theta):
        """
        Rotate vector v around unit vector k by angle theta (in radians).
        """
        # Compute the rotated vector using Rodrigues' formula
        v_rot = v * torch.cos(theta) + torch.linalg.cross(k, v) * torch.sin(theta) + k * torch.tensordot(k, v) * (1 - torch.cos(theta))
        
        return v_rot

    def calculateTransformationMatrix(self, grasp_axis, mid_points, approach_angles, grasp_translation=None):
        grasp_axis = grasp_axis / torch.norm(grasp_axis, dim=-1, keepdim=True)

        repeated_unit_vector = self.unit_vector.repeat(grasp_axis.shape[0], 1).to(grasp_axis.device)
        approach_axis = torch.linalg.cross(grasp_axis, repeated_unit_vector)
        approach_axis = approach_axis / torch.norm(approach_axis, dim=-1, keepdim=True)
        # print(approach_angles.shape, grasp_axis.shape, approach_axis.shape)
        approach_axis = self.rotate_vector(approach_axis, grasp_axis, approach_angles)
        approach_axis = approach_axis / torch.norm(approach_axis, dim=-1, keepdim=True)

        normal_axis = torch.linalg.cross(grasp_axis, approach_axis)
        normal_axis = normal_axis / torch.norm(normal_axis, dim=-1, keepdim=True)

        translation = mid_points - approach_axis * 1.12169998e-01
        # grasps = torch.eye(4).reshape(1, 4, 4).repeat(grasp_axis.shape[0], 1, 1)
        grasps = torch.zeros((grasp_axis.shape[0], 4, 4))

        grasps[:, :3, 0] = grasp_axis
        grasps[:, :3, 1] = normal_axis
        grasps[:, :3, 2] = approach_axis
        grasps[:, :3, 3] = translation
        grasps[:, 3, 3] = 1

        return grasps
    

if __name__ == "__main__":

    from tpp_dataset import TPPDataset
    from create_tpp_dataset import save_split_samples
    from torch_geometric.loader import DataLoader, DataListLoader
    from metrics import check_batch_grasp_success
    from torch_geometric.nn import DataParallel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = TppAngleNet().to(device)
        model = DataParallel(model)
    else:
        model = TppAngleNet()
        # model = DataParallel(model)
    # dataset_name = "tpp_seed"
    dataset_name = "tpp_effdict"
    train_paths, val_paths = save_split_samples('../data', 100, dataset_name)
    classification_criterion = nn.BCELoss()
    grasp_criterion = nn.MSELoss()
    # transform = RandomRotationTransform(rotation_range)
    
    train_dataset = TPPDataset(train_paths, transform=None, return_pair_dict=True)
    if device == "cuda":
        train_loader = DataListLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    num_success= 0
    for i, data in enumerate(train_loader):
        model.train()
        print(f"Batch: {i}/{len(train_loader)}")
        # pair_classification_out, pair_dot_product = model(data)
        grasp_outputs, selected_edge_idxs, mid_edge_pos, grasp_gt, num_valid_grasps, pair_classification_out_ij, mlp_out_ij = model(data)
        # print(grasp_outputs.shape, grasp_gt.shape, num_valid_grasps.shape, pair_classification_out_ij.shape, mlp_out_ij.shape)
        

        # classification_loss = classification_criterion(classification_output, approach_score_gt)
        # grasp_loss = grasp_criterion(grasp_outputs, grasp_gt)
        # grasp_outputs = grasp_outputs.reshape(-1, 4, 4).detach().numpy()
        # grasp_gt = grasp_gt.reshape(-1, 4, 4).detach().numpy()
        # num_success += check_batch_grasp_success(grasp_outputs, grasp_gt, 0.03, np.deg2rad(30))
        # # grasp_gt = torch.stack([sample.y for sample in data], dim=0)
        # # approach_gt = torch.stack([sample.approach_point_idx for sample in data], dim=0)
        # # loss = model.calculate_loss(grasp_gt, grasp_pred, approach_gt, approch_pred)
    
    # print(f"Success rate: {num_success / len(train_dataset)}")