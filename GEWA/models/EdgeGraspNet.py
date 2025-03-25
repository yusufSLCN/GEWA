import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  PPFConv,knn_graph,global_max_pool
from torch_geometric.nn import PointNetConv, radius
from torch.nn import Sequential, Linear, ReLU
import torch
from utils.metrics import get_binary_success_with_whole_gewa_dataset
from utils.visualize_acronym_dataset import visualize_grasps, visualize_edge_grasps
from sklearn.metrics import accuracy_score,balanced_accuracy_score
import warnings
warnings.filterwarnings("ignore")

class Classifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=(512,256,128)):
        super().__init__()
        self.head =  Sequential(Linear(in_channels, hidden_channels[0]),
                                ReLU(),
                                Linear(hidden_channels[0], hidden_channels[1]),
                                ReLU(),
                                Linear(hidden_channels[1], hidden_channels[2]),
                                ReLU(),
                                Linear(hidden_channels[2], 1))
        # self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.head(x)
        return x


class PointNetSimple(torch.nn.Module):
    def __init__(self, out_channels=(64,64,128)):
        super().__init__()
        torch.manual_seed(12345)

        in_channels = 3

        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        mlp1 = Sequential(Linear(in_channels + 3, out_channels[0]),
                          ReLU(),
                          Linear(out_channels[0], out_channels[0]))
        self.conv1 = PointNetConv(local_nn=mlp1)

        mlp2 = Sequential(Linear(out_channels[0] + 3, out_channels[1]),
                          ReLU(),
                          Linear(out_channels[1], out_channels[1]))
        self.conv2 = PointNetConv(local_nn=mlp2)

        mlp3 = Sequential(Linear(out_channels[1] + 3, out_channels[2]),
                          ReLU(),
                          Linear(out_channels[2], out_channels[2]))
        self.conv3 = PointNetConv(local_nn=mlp3)

    def forward(self, pos, batch=None, normal=None):
        # Compute the kNN graph:
        # Here, we need to pass the batch vector to the function call in order
        # to prevent creating edges between points of different examples.
        # We also add `loop=True` which will add self-loops to the graph in
        # order to preserve central point information.

        h = pos
        edge_index = knn_graph(pos, k = 16, batch=batch, loop=True)
        # 3. Start bipartite message passing.
        h1 = self.conv1(x=h, pos=pos, edge_index=edge_index)
        h1 = h1.relu()
        h2 = self.conv2(x=h1, pos=pos, edge_index=edge_index)
        #print('h', h.size())
        h2 = h2.relu()
        h3 = self.conv3(x=h2, pos=pos, edge_index=edge_index)
        h3 = h3.relu()
        # # 5. Classifier.
        return h1, h2, h3

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

    
class EdgeGrasp(nn.Module):
    def __init__(self, num_app_samples=32, num_contact_samples=32):
        """
        Initialize the EdgeGrasp model.

        Args:
            num_app_samples (int): Number of approach samples.
            num_contact_samples (int): Number of contact samples.
        """
        super(EdgeGrasp, self).__init__()
        self.num_app_samples = num_app_samples
        self.num_contact_samples = num_contact_samples
        self.local_emd_model = PointNetSimple(out_channels=(32, 64, 128))
        self.global_emd_model = GlobalEmdModel(input_c=32+64+128, inter_c=(256,512,512),output_c=1024)
        self.classifier = Classifier(in_channels=32+64+128+1024, hidden_channels=(512, 256, 128))

        self.gripper_width = 0.082
        # self.gripper_half_width = self.gripper_width / 2
        self.gripper_depth = 0.072-0.007 #edge graspnet values
        # self.gripper_depth = 1.12169998e-01 - 6.59999996e-02
        # self.gripper_depth = 6.59999996e-02

        # self.success_mse = nn.MSELoss()
        self.success_mse = nn.BCEWithLogitsLoss()


    def forward(self, data):
        # Todo get the local emd for every point in the batch
        # balls setup
        pos, batch, normals = data.pos, data.batch, data.normals
        
        batch_size = len(data)
        # pos = pos[torch.randint(0, pos.size(0), (self.sample_num,))]
        batch_grasp_pred = torch.zeros((batch_size, self.num_app_samples, self.num_contact_samples, 4, 4))
        batch_success_pred = torch.zeros((batch_size, self.num_app_samples, self.num_contact_samples))
        batch_success_gt = torch.zeros((batch_size, self.num_app_samples, self.num_contact_samples))
        batch_approach_points = torch.zeros((batch_size, self.num_app_samples, 3))
        batch
        loss = 0
        for obj_i in range(batch_size):
            obj_mask = batch == obj_i
            obj_pos = pos[obj_mask]
            obj_normals = normals[obj_mask]

            #sample random points from the point cloud
            appr_point_idx = torch.randint(0, obj_pos.shape[0], (self.num_app_samples,))
            appr_points = obj_pos[appr_point_idx]
            batch_approach_points[obj_i] = appr_points
            radius_p_batch_index = radius(obj_pos, appr_points, r=0.038, max_num_neighbors=1024)
            radius_p_index = radius_p_batch_index[1, :]
            radius_p_batch = radius_p_batch_index[0, :]

            #do a minibatch for each sampled approach point
            for i in range(self.num_app_samples):
                ball_mask = radius_p_batch == i
                ball_idxs = radius_p_index[ball_mask]
                ball_pos = obj_pos[ball_idxs]
                ball_normals = obj_normals[ball_idxs]
                

                batch_i = torch.zeros(ball_pos.size(0), dtype=torch.long).to(ball_pos.device)
                f1, f2, features = self.local_emd_model(pos=ball_pos, batch=batch_i)
                contact_emd = torch.cat((f1,f2,features),dim=1)
                global_emd = self.global_emd_model(contact_emd, batch_i)
                repeated_global_emd = torch.cat([global_emd[i,:].repeat((batch_i==i).sum(),1) for i in range(len(global_emd))],dim=0)
                edge_features = torch.cat((contact_emd, repeated_global_emd),dim=-1)
                classifier_out = self.classifier(edge_features)

                #sample contact points
                contact_point_idx = torch.randint(0, ball_pos.shape[0], (self.num_contact_samples,))
                contact_points = ball_pos[contact_point_idx]
                contact_normals = ball_normals[contact_point_idx]
                approach_point = appr_points[i]

                #calculate the transformation matrix from the contact normals, contactpoints and approach point
                grasp_pred = self.calculate_transformation(approach_point, contact_points, contact_normals)
                success_pred = classifier_out[contact_point_idx].squeeze()

                batch_grasp_pred[obj_i, i] = grasp_pred
                batch_success_pred[obj_i, i] = success_pred

                #create the ground truth by checking the success of the grasps
                grasp_gt_path = data[obj_i].sample_info['grasps']
                mean = data[obj_i].sample_info['mean']
                grasp_pred = grasp_pred.cpu().detach().reshape(self.num_contact_samples, 1, 4, 4).numpy()
                binary_success_gt = get_binary_success_with_whole_gewa_dataset(grasp_pred, 0.03, np.deg2rad(30), grasp_gt_path, mean)
                # print(np.sum(binary_success_gt))
                binary_success_gt = torch.tensor(binary_success_gt, dtype=torch.float32).to(success_pred.device)
                batch_success_gt[obj_i, i] = binary_success_gt

                # pos_grasps_mask_gt = binary_success_gt > 0
                # pos_grasp_count = torch.sum(pos_grasps_mask_gt)
                # neg_pos_ratio = binary_success_gt.shape[0] - pos_grasp_count / pos_grasp_count
                # weighted_success_mse = nn.BCEWithLogitsLoss()

                # print(binary_success_gt)
                # print(success_pred)
                # loss += weighted_success_mse(success_pred, binary_success_gt)
                loss += self.success_mse(success_pred, binary_success_gt)
                
                # vis_num = 5
                # trans_m = trans_m[:vis_num]
                # contact_points = contact_points[:vis_num]
                # visualize_edge_grasps(obj_pos, trans_m, approach_point=approach_point, contact_points=contact_points, contact_normals=contact_normals)

        return loss, batch_grasp_pred, batch_success_pred, batch_success_gt, batch_approach_points

    def calculate_transformation(self, appr_point, contact_pts, contact_normals):
        relative_pos =  appr_point - contact_pts
        relative_pos_norm = F.normalize(relative_pos, p=2, dim=1)
        contact_normals = F.normalize(contact_normals, p=2, dim=1)

        x_axis = torch.cross(contact_normals, relative_pos_norm, dim=1)
        x_axis = F.normalize(x_axis, p=2, dim=1)

        approach_axis = torch.cross(x_axis, contact_normals, dim=1)
        approach_axis = F.normalize(approach_axis, p=2, dim=1)
        approach_axis = -approach_axis

        rotation_m = torch.stack([contact_normals, x_axis, approach_axis], dim=2)

        dot_product =  -torch.sum(relative_pos * approach_axis, dim=1)
        translation_shift_from_approach = self.gripper_depth + dot_product
        translation_shift_from_approach = translation_shift_from_approach.unsqueeze(1)
        gripper_position = appr_point - translation_shift_from_approach * approach_axis

        trans_m = torch.eye(4).repeat(len(contact_pts), 1, 1)
        trans_m[:, :3, :3] = rotation_m
        trans_m[:, :3, 3] = gripper_position

        return trans_m


if __name__ == "__main__":

    from dataset.tpp_dataset import TPPDataset
    from dataset.create_tpp_dataset import save_contactnet_split_samples
    from torch_geometric.loader import DataLoader
    from utils.metrics import check_batch_grasp_success
    import numpy as np
    import torch.nn as nn
    from torcheval.metrics.functional.classification import binary_accuracy

    model = EdgeGrasp() 
    #count the number of parameters
    print("Number of params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_paths, val_paths = save_contactnet_split_samples('../data', num_mesh=1200, dataset_name="tpp_effdict_nomean_wnormals")
    
    dataset = TPPDataset(val_paths, return_pair_dict=False, normalize=True, return_normals=True)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    classification_criterion = nn.BCELoss()
    # transform = RandomRotationTransform(rotation_range)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for data in data_loader:
        optimizer.zero_grad()
        loss, batch_grasp_pred, batch_success_pred, batch_success_gt, batch_approach_points = model(data)

        batch_success_pred = batch_success_pred.flatten()
        batch_success_gt = batch_success_gt.flatten()
        accuracy = binary_accuracy(batch_success_pred, batch_success_gt)
        print("Acc: ", accuracy)
        print(loss)
        loss.backward()
        optimizer.step()
        # print(classifier_out, grasp_outputs, approach_points, grasp_gt)
