import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import  PPFConv,knn_graph,global_max_pool
from torch_geometric.nn import PointNetConv
from torch.nn import Sequential, Linear, ReLU
import torch
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
        #self.sigmoid = torch.nn.Sigmoid()
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
    
class EdgeGrasp(nn.Module):
    def __init__(self, sample_num=32, approach_feat_dim=64, grasp_dim=16):
        super(EdgeGrasp, self).__init__()
        self.sample_num = sample_num
        self.local_emd_model = PointNetSimple(out_channels=(32, 64, 128))
        self.global_emd_model = GlobalEmdModel(input_c=32+64+128, inter_c=(256,512,512),output_c=1024)
        self.classifier = Classifier(in_channels=32+64+128+1024, hidden_channels=(512, 256, 128))
        self.grasp_predictor = GraspPosePredictor(global_feat_dim=32+64+128+1024, approach_feat_dim=approach_feat_dim, grasp_dim=grasp_dim)


    def forward(self, data):
        # Todo get the local emd for every point in the batch
        # balls setup
        pos, batch = data.pos, data.batch

        f1, f2, features = self.local_emd_model(pos=pos, batch=batch)



        des_emd = torch.cat((f1,f2,features),dim=1)
 
        global_emd = self.global_emd_model(des_emd, batch)
        repeated_global_emd = torch.cat([global_emd[i,:].repeat((batch==i).sum(),1) for i in range(len(global_emd))],dim=0)
        model_embed = torch.cat((des_emd,repeated_global_emd),dim=-1)

        classifier_out = self.classifier(model_embed)
        grasp_outputs = 0
        approach_points = 0
        grasp_gt = 0

        return classifier_out, grasp_outputs, approach_points, grasp_gt


if __name__ == "__main__":

    from gewa_dataset import GewaDataset
    from create_gewa_dataset import save_split_samples
    from torch_geometric.loader import DataLoader
    from metrics import check_batch_grasp_success
    import numpy as np
    import torch.nn as nn

    model = EdgeGrasp() 
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
        # approach_score_gt = data.approach_scores
        # approach_score_gt = (approach_score_gt > 0).float()
        # # print(approach_score_gt.shape, grasp_target.shape)
        # classification_loss = classification_criterion(classification_output, approach_score_gt)
        # grasp_loss = grasp_criterion(grasp_outputs, grasp_gt)
        # grasp_outputs = grasp_outputs.reshape(-1, 4, 4).detach().numpy()
        # grasp_gt = grasp_gt.reshape(-1, 4, 4).detach().numpy()
        # num_success += check_batch_grasp_success(grasp_outputs, grasp_gt, 0.03, np.deg2rad(30))
        # grasp_gt = torch.stack([sample.y for sample in data], dim=0)
        # approach_gt = torch.stack([sample.approach_point_idx for sample in data], dim=0)
        # loss = model.calculate_loss(grasp_gt, grasp_pred, approach_gt, approch_pred)
    
    print(f"Success rate: {num_success / len(train_dataset)}")