import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, global_mean_pool
import numpy as np

class TppNet(nn.Module):
    def __init__(self, grasp_dim=9, k=8, num_grasp_sample=100, num_points=1000, max_num_grasps=10, only_classifier=False):
        super(TppNet, self).__init__()
        
        self.num_grasp_sample = num_grasp_sample
        self.grap_dim = grasp_dim
        self.k = k
        self.multi_gpu = False
        self.num_points = num_points
        self.max_num_grasps = max_num_grasps
        self.only_classifier = only_classifier
        self.triu = torch.triu_indices(num_points, num_points, offset=1)

        self.num_pairs = self.triu.shape[1]
        
        self.conv1 = DynamicEdgeConv(MLP([6, 16, 32]), k=self.k, aggr='max')
        self.conv2 = DynamicEdgeConv(MLP([64, 64, 128]), k=self.k, aggr='max')
        self.conv3 = DynamicEdgeConv(MLP([256, 256, 512]), k=self.k, aggr='max')
        # self.conv4 = DynamicEdgeConv(MLP([1024, 1024, 2048]), k=self.k, aggr='max')
        # self.conv5 = DynamicEdgeConv(MLP([256, 256, 512]), k=self.k, aggr='max')
        
        self.point_feat_dim = 32
        self.shared_mlp = nn.Sequential(
            MLP([512 + 128 + 32, 256, self.point_feat_dim]),
            # nn.Linear(128, self.point_feat_dim)
        ) 

        # self.edge_classifier = nn.Sequential(
        #     nn.Linear(self.point_feat_dim * 2, self.point_feat_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.point_feat_dim, 1)
        # )

        self.edge_classifier = nn.Sequential(
            nn.Linear(self.point_feat_dim * 3, self.point_feat_dim),
            nn.ReLU(),
            nn.Linear(self.point_feat_dim, 1)
        )
        
        # Grasp prediction head
        self.grasp_head = nn.Sequential(
            nn.Linear(self.point_feat_dim * 3, self.point_feat_dim),
            nn.ReLU(),
            nn.Linear(self.point_feat_dim, self.grap_dim)
        )

    def forward(self, data, temperature=1.0):
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
        # edge_feature_ji = torch.cat([feat_j, feat_i, global_embedding], dim=-1)
        # mlp_out_ji = self.edge_classifier(edge_feature_ji)
        # mlp_out_ji = mlp_out_ji.squeeze(-1)
        # pair_classification_out_ji = torch.sigmoid(mlp_out_ji)

        # return pair_classification_out_ij, mlp_out_ij
        #------------------------------------------------------
        # shared_features = shared_features.reshape(-1, self.num_points, self.point_feat_dim)
        # feat_i = shared_features[:, self.triu[0], :]
        # feat_j = shared_features[:, self.triu[1], :]
        # edge_feature_ij = torch.cat([feat_i, feat_j], dim=-1)
        # mlp_out_ij = self.edge_classifier(edge_feature_ij)
        # mlp_out_ij = mlp_out_ij.squeeze(-1)
        # pair_classification_out_ij = torch.sigmoid(mlp_out_ij)

        # edge_feature_ji = torch.cat([feat_j, feat_i], dim=-1)
        # mlp_out_ji = self.edge_classifier(edge_feature_ji)
        # mlp_out_ji = mlp_out_ji.squeeze(-1)
        # pair_classification_out_ji = torch.sigmoid(mlp_out_ji)

        # return pair_classification_out_ij, mlp_out_ij, pair_classification_out_ji, mlp_out_ji
        #------------------------------------------------------
        # shared_features = shared_features.reshape(-1, 1000, self.point_feat_dim)
        # dot_product = torch.matmul(shared_features, shared_features.transpose(1, 2))
        # pair_dot_product = dot_product[:, self.triu[0], self.triu[1]]
        # # print(pair_dot_product.shape)
        # # out_features = self.dot_product_head(pair_dot_product)

        # pair_classification_out = torch.sigmoid(pair_dot_product)

        # return pair_classification_out, pair_dot_product
    
        #------------------------------------------------------

        # global_features = global_mean_pool(shared_features, batch)
        # global_features = global_features.reshape(-1, 1, self.point_feat_dim)
        # global_features = global_features.repeat(1, 1000, 1)
        # shared_features = shared_features.reshape(-1, 1000, self.point_feat_dim)

        # combined_features = torch.cat([shared_features, global_features], dim=2)
        # dot_product = torch.matmul(combined_features, combined_features.transpose(1, 2))
        # pair_dot_product = dot_product[:, self.triu[0], self.triu[1]]
        # # print(pair_dot_product.shape)
        # # out_features = self.dot_product_head(pair_dot_product)
        # pair_classification_out = torch.sigmoid(pair_dot_product)

        # return pair_classification_out, pair_dot_product


        #------------------------------------------------------
        
        selected_edge_idxs = []
        selected_edge_features = []
        grasp_gt = np.zeros((self.num_grasp_sample, self.max_num_grasps, 4, 4))
        mid_edge_pos = []

        # print(data.y[list(data.y.keys())[0]])
        # print(data.pair_scores.shape)
        edge_scores = data.pair_scores
        edge_scores = edge_scores.reshape(-1, self.num_pairs)
        pos = pos.reshape(-1, self.num_points, 3)
        grasp_gt = np.zeros((pos.shape[0], self.num_grasp_sample, self.max_num_grasps, 4, 4))
        num_valid_grasps = np.zeros((pos.shape[0], self.num_grasp_sample))
        # print(edge_scores.shape)
        # print(data.y)
        for i in range(edge_scores.shape[0]):  # Iterate over each point cloud in the batch
            sample_pos = pos[i]
            # sample_num_grasps = data.num_grasps[mask]
            
            # Multinomial sampling for point selection
            if self.training:
                sample_edge_scores = edge_scores[i]
                with_replacement = torch.sum(sample_edge_scores > 0.5) < self.num_grasp_sample
                edge_index = torch.multinomial(sample_edge_scores, num_samples=self.num_grasp_sample,
                                                replacement=with_replacement.item())
            else:
                sample_edge_prob = pair_classification_out_ij[i]
                with_replacement = torch.sum(sample_edge_prob > 0.5) < self.num_grasp_sample
                edge_index = torch.multinomial(sample_edge_prob, num_samples=self.num_grasp_sample, 
                                                replacement=with_replacement.item())
            
            # print("find edge indxs")
            edge_index = edge_index.to("cpu")
            selected_edge_node1 = self.triu[0][edge_index]
            selected_edge_node2 = self.triu[1][edge_index]
            selected_edge_idxs.append(edge_index)
            sample_edge_grasps = []
            # print(data.y)
            # print("selection loop start")
            for edge_j, (point1_idx, point2_idx) in enumerate(zip(selected_edge_node1, selected_edge_node2)):
                point1_idx = point1_idx.item()
                point2_idx = point2_idx.item()
                grasp_key = frozenset((point1_idx, point2_idx))
                if grasp_key in data.y[i][0]:
                    edge_grasps = data.y[i][0][grasp_key]
                    # if len(edge_grasps) > 1:
                    #     print(len(edge_grasps))
                    # print(len(data.y[grasp_key]))
                    # print(edge_grasps[0].shape)
                    if len(edge_grasps) > self.max_num_grasps:
                        edge_grasps = edge_grasps[:self.max_num_grasps]

                    if len(edge_grasps) > 0:
                        grasp_gt[i, edge_j, :len(edge_grasps)] = edge_grasps
                        num_valid_grasps[i, edge_j] = len(edge_grasps)

                # sample_edge_grasps.append(edge_grasps)
                mid_edge_pos.append((sample_pos[point1_idx] + sample_pos[point2_idx]) / 2)
            

            selected_edge_features.append(edge_feature_ij[i, edge_index])

        selected_edge_features = torch.stack(selected_edge_features)
        
        grasp_outputs = self.grasp_head(selected_edge_features)
        mid_edge_pos = torch.stack(mid_edge_pos)
        selected_edge_idxs = torch.stack(selected_edge_idxs)
        grasp_gt = torch.tensor(grasp_gt, dtype=torch.float32)
        num_valid_grasps = torch.tensor(num_valid_grasps, dtype=torch.int8)
        # print("calculate grasps matrix")
        # grasp_outputs = self.grasp_head(grasp_features)
        if self.grap_dim != 16:
            grasp_outputs = grasp_outputs.reshape(-1, self.grap_dim)
            grasp_outputs = self.calculateTransformationMatrix(grasp_outputs, mid_edge_pos)
            grasp_outputs = grasp_outputs.view(-1, 16)
        else:
            grasp_outputs = grasp_outputs.reshape(-1, 4, 4)
            grasp_outputs[:, :3, 3] + mid_edge_pos
            grasp_outputs = grasp_outputs.view(-1, 16)

        # print("return")
        selected_edge_idxs = selected_edge_idxs.to(grasp_outputs.device)
        grasp_gt = grasp_gt.to(grasp_outputs.device)
        num_valid_grasps = num_valid_grasps.to(grasp_outputs.device)
        # print(grasp_outputs.device, selected_edge_idxs.device,
        #        mid_edge_pos.device, grasp_gt.device, num_valid_grasps.device, pair_classification_out_ij.device, mlp_out_ij.device)


        return grasp_outputs, selected_edge_idxs, mid_edge_pos, grasp_gt, num_valid_grasps, pair_classification_out_ij, mlp_out_ij

    
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

    from tpp_dataset import TPPDataset
    from create_tpp_dataset import save_split_samples
    from torch_geometric.loader import DataLoader, DataListLoader
    from metrics import check_batch_grasp_success
    from torch_geometric.nn import DataParallel

    model = TppNet().to("cuda")
    model = DataParallel(model)
    # dataset_name = "tpp_seed"
    dataset_name = "tpp_effdict"
    train_paths, val_paths = save_split_samples('../data', 100, dataset_name)
    classification_criterion = nn.BCELoss()
    grasp_criterion = nn.MSELoss()
    # transform = RandomRotationTransform(rotation_range)
    
    train_dataset = TPPDataset(train_paths, transform=None, return_pair_dict=True)
    train_loader = DataListLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
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