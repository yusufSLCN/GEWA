import torch
"""
TppBallNet: An extension of the TPPNet that extracts local features around the selected edge.
Classes:
    GlobalEmdModel(nn.Module):
        A neural network model for extracting global embeddings from positional embeddings.
        Methods:
            __init__(self, input_c=128, inter_c=(256, 512, 512), output_c=512):
                Initializes the GlobalEmdModel with the given input, intermediate, and output dimensions.
            forward(self, pos_emd, radius_p_batch):
                Forward pass to compute the global embedding from positional embeddings and batch indices.
    TppBallNet(nn.Module):
        A neural network model for grasp prediction using dynamic edge convolutions and global embeddings.
        Methods:
            __init__(self, grasp_dim=7, k=8, num_grasp_sample=100, num_points=1000, max_num_grasps=10, only_classifier=False,
                Initializes the TppBallNet with the given parameters for grasp prediction.
            forward(self, data):
                Forward pass to compute grasp predictions and edge classifications from input data.
            calculateTransformationMatrix(self, grasp, mid_points):
                Calculates the transformation matrix for the given grasp and midpoints.
"""
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, MLP, global_max_pool, radius
import numpy as np
from torch.nn import Sequential, Linear, ReLU

class GlobalEmdModel(torch.nn.Module):
    def __init__(self,input_c = 128, inter_c=(256,512,512), output_c=512):
        super().__init__()
        self.mlp1 = Sequential(Linear(input_c, inter_c[0]), ReLU(), Linear(inter_c[0], inter_c[1]), ReLU(), Linear(inter_c[1], inter_c[2]),)
        self.mlp2 = Sequential(Linear(input_c+inter_c[2], output_c), ReLU(), Linear(output_c, output_c))
    def forward(self,pos_emd,radius_p_batch):
        global_emd = self.mlp1(pos_emd)
        global_emd = global_max_pool(global_emd, radius_p_batch)
        global_emd = torch.cat([global_emd[i,:].repeat((radius_p_batch==i).sum(),1) for i in range(len(global_emd))],dim=0)
        global_emd = torch.cat((pos_emd,global_emd),dim=-1)
        global_emd = self.mlp2(global_emd)
        global_emd = global_max_pool(global_emd, radius_p_batch)
        return global_emd
    

class TppBallNet(nn.Module):
    def __init__(self, grasp_dim=7, k=8, num_grasp_sample=100, num_points=1000, max_num_grasps=10, only_classifier=False,
                  sort_by_score=True, with_normals=False, normalize=False, topk=10):
        """
        Initializes the TppBallNet with the given parameters for grasp prediction.
        
        Args:
            grasp_dim (int): Dimension of the grasp representation.
            k (int): Number of nearest neighbors for dynamic edge convolution.
            num_grasp_sample (int): Number of grasp samples.
            num_points (int): Number of points in the point cloud.
            max_num_grasps (int): Maximum number of grasps.
            only_classifier (bool): If True, only the edge classifier is used.
            sort_by_score (bool): If True, sort edges by score.
            with_normals (bool): If True, use normals in the input.
            normalize (bool): If True, normalize the grasp.
            topk (int): Number of top edges to consider.
        """
        super(TppBallNet, self).__init__()
        
        self.num_grasp_sample = num_grasp_sample
        self.grap_dim = grasp_dim
        self.k = k
        self.multi_gpu = False
        self.num_points = num_points
        self.max_num_grasps = max_num_grasps
        self.only_classifier = only_classifier
        self.sort_by_score = sort_by_score
        self.normalize = normalize
        self.topk = topk

        self.with_normals = with_normals
        self.triu = torch.triu_indices(num_points, num_points, offset=1)

        self.num_pairs = self.triu.shape[1]
        
        input_dim = 12 if self.with_normals else 6
        self.conv1 = DynamicEdgeConv(MLP([input_dim, 16, 32]), k=self.k, aggr='max')
        self.conv2 = DynamicEdgeConv(MLP([64, 64, 128]), k=self.k, aggr='max')
        self.conv3 = DynamicEdgeConv(MLP([256, 256, 512]), k=self.k, aggr='max')
        # self.conv4 = DynamicEdgeConv(MLP([1024, 1024, 2048]), k=self.k, aggr='max')
        # self.conv5 = DynamicEdgeConv(MLP([256, 256, 512]), k=self.k, aggr='max')
        
        self.point_feat_dim = 128
        self.shared_mlp = nn.Sequential(
            MLP([512 + 128 + 32, 256, self.point_feat_dim]),
            # nn.Linear(128, self.point_feat_dim)
        ) 

        self.edge_classifier = nn.Sequential(
            nn.Linear(self.point_feat_dim * 3, self.point_feat_dim),
            nn.ReLU(),
            nn.Linear(self.point_feat_dim, 1)
        )

        self.ball_global_emd_dim = 256
        #extact ball global features
        self.ball_global_encoder = GlobalEmdModel(input_c=self.point_feat_dim, inter_c=(self.point_feat_dim,
                                                                                        self.ball_global_emd_dim,
                                                                                        self.ball_global_emd_dim),
                                                                                          output_c=self.ball_global_emd_dim)
        
        # Grasp prediction head
        self.grasp_head = nn.Sequential(
            nn.Linear(self.point_feat_dim * 3 + self.ball_global_emd_dim, self.point_feat_dim + self.ball_global_emd_dim),
            nn.ReLU(),
            nn.Linear(self.point_feat_dim + self.ball_global_emd_dim, self.point_feat_dim),
            nn.ReLU(),
            nn.Linear(self.point_feat_dim, self.grap_dim)
        )

    def forward(self, data):
        pos, batch = data.pos, data.batch
        if self.with_normals:
            normals = data.normals
            input = torch.cat((pos, normals), dim=1)
            # print(pos.shape, normals.shape)
        else:
            input = pos
            
        x1 = self.conv1(input, batch)
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
        ball_global_features = []
        mid_edge_pos = []
        grasp_touch_points = []
        grasp_axises = []
        edge_lengths = []

        edge_scores = data.pair_scores
        edge_scores = edge_scores.reshape(-1, self.num_pairs)
        pos = pos.reshape(-1, self.num_points, 3)

        if self.training:
            num_valid_grasps = np.zeros((pos.shape[0], self.num_grasp_sample))
            grasp_gt = np.zeros((pos.shape[0], self.num_grasp_sample, self.max_num_grasps, 4, 4))
        else:
            num_valid_grasps = np.zeros((pos.shape[0], self.topk))
            grasp_gt = np.zeros((pos.shape[0], self.topk, self.max_num_grasps, 4, 4))


        # iterate over each mesh in the batch and sample graspable edges
        for i in range(edge_scores.shape[0]):  
            sample_pos = pos[i]
            batch_mid_edge_pos = []
            # Multinomial sampling for point selection
            if self.training:
                sample_edge_scores = edge_scores[i]
                sample_edge_prob = (sample_edge_scores > 0.5).float()
                with_replacement = torch.sum(sample_edge_prob) < self.num_grasp_sample
                edge_index = torch.multinomial(sample_edge_prob, num_samples=self.num_grasp_sample,
                                                replacement=with_replacement.item())
            else:
                sample_edge_prob = pair_classification_out_ij[i]
                if self.sort_by_score:
                    sorted_score = torch.argsort(sample_edge_prob, descending=True)
                    edge_index = sorted_score[:self.topk]
                else:
                    pos_pair_count = torch.sum(sample_edge_prob > 0.5)
                    if pos_pair_count > 0:
                        sample_edge_prob[sample_edge_prob < 0.5] = 0
                    with_replacement = pos_pair_count < self.topk
                    edge_index = torch.multinomial(sample_edge_prob, num_samples=self.topk, 
                                                    replacement=with_replacement.item())
            
            edge_index = edge_index.to("cpu")
            selected_edge_node1 = self.triu[0][edge_index]
            selected_edge_node2 = self.triu[1][edge_index]
            selected_edge_idxs.append(edge_index)
            selected_edge_features.append(edge_feature_ij[i, edge_index])


            radius_p_index = []
            radius_p_batch = []
            # print("selection loop start")
            for edge_j, (point1_idx, point2_idx) in enumerate(zip(selected_edge_node1, selected_edge_node2)):
                point1_idx = point1_idx.item()
                point2_idx = point2_idx.item()
                grasp_key = frozenset((point1_idx, point2_idx))
                sample_grasp_dict = data.y[i][0]
                if grasp_key in sample_grasp_dict:
                    edge_grasps = sample_grasp_dict[grasp_key]
                    if self.normalize:
                        mean = data.sample_info["mean"][i]
                        edge_grasps[:, :3, 3] -= mean

                    if len(edge_grasps) > self.max_num_grasps:
                        edge_grasps = edge_grasps[:self.max_num_grasps]

                    if len(edge_grasps) > 0:
                        # edge_grasps = torch.eye(4).reshape(1, 4, 4).repeat(self.max_num_grasps, 1, 1)
                        # grasp_gt[i, edge_j] = edge_grasps
                        grasp_gt[i, edge_j, :len(edge_grasps)] = edge_grasps
                        num_valid_grasps[i, edge_j] = len(edge_grasps)

                grasp_axis = sample_pos[point2_idx] - sample_pos[point1_idx]
                grasp_axises.append(grasp_axis)
                edge_lenght = torch.linalg.vector_norm(grasp_axis, dim=-1)
                edge_lengths.append(edge_lenght)
                mid_point = (sample_pos[point1_idx] + sample_pos[point2_idx]) / 2
                mid_edge_pos.append(mid_point)
                touch_point_pair = torch.hstack([sample_pos[point1_idx], sample_pos[point2_idx]])
                grasp_touch_points.append(touch_point_pair)


                #extract ball global features
                # edge_half_length = edge_lengths[edge_j]
                # crop_center = mid_edge_pos[edge_j] 
                if edge_lenght < 0.01:
                    edge_lenght = 0.01
                # print(sample_pos.shape)
                radius_p_batch_index = radius(sample_pos, mid_point, r=edge_lenght/2, max_num_neighbors=64)
                cropped_idxs = radius_p_batch_index[1, :]
                if cropped_idxs.shape[0] == 0:
                    cropped_idxs = torch.tensor([point1_idx, point2_idx])
                # distances = torch.linalg.vector_norm(sample_pos - mid_point, dim=-1)
                # cropped_idxs = distances < (edge_lenght )
                # cropped_idxs = torch.where(cropped_idxs)[0]
                # print(cropped_idxs.shape, edge_lenght, mid_point)
                radius_p_index.append(cropped_idxs)
                radius_p_batch.append(torch.zeros_like(cropped_idxs) + edge_j)

            radius_p_index = torch.cat(radius_p_index)
            radius_p_batch = torch.cat(radius_p_batch)
            # print(radius_p_index.shape, torch.max(radius_p_batch))
            ball_shared_features = shared_features[i, radius_p_index]
            ball_global_emd = self.ball_global_encoder(ball_shared_features, radius_p_batch)
            ball_global_features.append(ball_global_emd)

        # prepare grasp features
        grasp_axises = torch.stack(grasp_axises)
        grasp_axises = grasp_axises / torch.norm(grasp_axises, dim=-1, keepdim=True)
        selected_edge_features = torch.stack(selected_edge_features)
        mid_edge_pos = torch.stack(mid_edge_pos)
        ball_global_emd = torch.stack(ball_global_features)
        
        # predict grasps
        combined_features = torch.cat([selected_edge_features, ball_global_emd], dim=-1)
        grasp_outputs = self.grasp_head(combined_features)
        selected_edge_idxs = torch.stack(selected_edge_idxs)
        grasp_gt = torch.tensor(grasp_gt, dtype=torch.float32)
        num_valid_grasps = torch.tensor(num_valid_grasps, dtype=torch.int8)

        #calculate transformation matrix
        grasp_outputs = grasp_outputs.reshape(-1, self.grap_dim)
        grasp_outputs = self.calculateTransformationMatrix(grasp_outputs, mid_edge_pos)
        grasp_outputs = grasp_outputs.view(-1, 16)

        selected_edge_idxs = selected_edge_idxs.to(grasp_outputs.device)
        grasp_gt = grasp_gt.to(grasp_outputs.device)
        num_valid_grasps = num_valid_grasps.to(grasp_outputs.device)
        # print(grasp_outputs.device, selected_edge_idxs.device,
        #        mid_edge_pos.device, grasp_gt.device, num_valid_grasps.device, pair_classification_out_ij.device, mlp_out_ij.device)


        return grasp_outputs, selected_edge_idxs, mid_edge_pos, grasp_axises, grasp_gt, num_valid_grasps, pair_classification_out_ij, mlp_out_ij

    
    def calculateTransformationMatrix(self, grasp, mid_points):
        # translation = grasp[:, :3] + mid_points
        r1 = grasp[:, :3]
        r2 = grasp[:, 3:6]
        #orthogonalize the rotation vectors
        r1 = r1 / torch.norm(r1, dim=1, keepdim=True)
        r2 = r2 - torch.sum(r1 * r2, dim=1, keepdim=True) * r1
        r2 = r2 / torch.norm(r2, dim=1, keepdim=True)
        r3 = torch.cross(r1, r2, dim=1)
        r3 = r3 / torch.norm(r3, dim=1, keepdim=True)
        #create the rotation matrix
        r = torch.stack([r1, r2, r3], dim=2)

        gaxis_translation_scale = grasp[:, -1].reshape(-1, 1)
        grasp_axis_trans_shift = gaxis_translation_scale * r1
        translation = mid_points - r3 * 1.12169998e-01 + grasp_axis_trans_shift
        #create 4x4 transformation matrix for each 
        trans_m = torch.eye(4).repeat(len(grasp), 1, 1).to(grasp.device)
        trans_m[:,:3, :3] = r
        trans_m[:, :3, 3] = translation
        return trans_m
if __name__ == "__main__":

    from dataset.tpp_dataset import TPPDataset
    from dataset.create_tpp_dataset import save_contactnet_split_samples
    from torch_geometric.loader import DataLoader, DataListLoader
    from utils.metrics import check_batch_grasp_success
    from torch_geometric.nn import DataParallel

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    if device == "cuda":
        model = TppBallNet(normalize=True).to(device)
        model = DataParallel(model)
    else:
        model = TppBallNet(normalize=True)

    #count the parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {pytorch_total_params}")

    # model = DataParallel(model)
    # dataset_name = "tpp_seed"
    dataset_name = "tpp_effdict_nomean_wnormals"
    train_paths, val_paths = save_contactnet_split_samples('../data', 1200, dataset_name)
    classification_criterion = nn.BCELoss()
    grasp_criterion = nn.MSELoss()
    # transform = RandomRotationTransform(rotation_range)
    
    train_dataset = TPPDataset(train_paths, transform=None, return_pair_dict=True, normalize=True)
    if device == "cuda":
        train_loader = DataListLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    num_success= 0
    for i, data in enumerate(train_loader):
        model.train()
        print(f"Batch: {i}/{len(train_loader)}")
        # pair_classification_out, pair_dot_product = model(data)
        grasp_outputs, selected_edge_idxs, mid_edge_pos, grasp_axises, grasp_gt, num_valid_grasps, pair_classification_out_ij, mlp_out_ij = model(data)
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