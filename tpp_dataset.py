import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from create_tpp_dataset import save_split_samples
import numpy as np 

    
class TPPDataset(Dataset):
    def __init__(self, data, transform=None, normalize_points=True, max_grasp_perpoint=20):
        self.data = data
        self.transform = transform
        self.normalize_points = normalize_points
        self.device = "cpu"
        self.max_grasp_perpoint = max_grasp_perpoint
        self.triu_indices = np.triu_indices(1000, k=1)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Process the sample if needed
        point_cloud = sample.point_cloud
        touch_pair_score_matrix = sample.touch_pair_scores
        pair_grasps_dict = sample.pair_grasps 
        sample_info = sample.info
  
        # convert to torch tensors
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        pair_scores = torch.zeros((len(self.triu_indices[0])), dtype=torch.int16)

        pair_grasps = torch.zeros((len(self.triu_indices[0]), self.max_grasp_perpoint, 4, 4), dtype=torch.float32)
        for i in range(len(self.triu_indices[0])):
            num_pair_grasps = 0
            num_pair_grasps = touch_pair_score_matrix[self.triu_indices[0][i], self.triu_indices[1][i]]
            num_pair_grasps += touch_pair_score_matrix[self.triu_indices[1][i], self.triu_indices[0][i]]
            pair_scores[i] = num_pair_grasps

            key = frozenset((self.triu_indices[0][i], self.triu_indices[1][i]))
            if key in pair_grasps_dict:
                grasps = np.array(pair_grasps_dict[key])
                grasps = torch.tensor(grasps)
                if grasps.shape[0] > 0:
                    pair_grasps[i, :grasps.shape[0]] = grasps
            
        # contact_points = np.array([[point_grasps[i][1], point_grasps[i][2]]  for i in range(len(point_grasps))])
        # contact_points = torch.tensor(contact_points, dtype=torch.float32)
        # num_grasps = torch.tensor([len(point_grasps[i]) for i in range(len(point_grasps))])
        # num_grasps[num_grasps > self.max_grasp_perpoint] = self.max_grasp_perpoint
        # grasps = np.zeros((len(point_grasps), self.max_grasp_perpoint, 4, 4))
        # for i in range(len(point_grasps)):
        #     for j in range(self.max_grasp_perpoint):
        #         if j < len(point_grasps[i]):
        #             grasps[i, j] = point_grasps[i][j]
        #         else:
        #             break

        # grasps = np.array([point_grasps[i][0] for i in range(len(point_grasps))])

        #normalize the vertices
        if self.normalize_points:
            mean = torch.mean(point_cloud, axis=0)
            point_cloud = point_cloud - mean
            sample_info['mean'] = mean
            pair_grasps[:, :, :3, 3] = pair_grasps[:, :, :3, 3] - mean

        # success = torch.tensor(success)
        data = Data(x=point_cloud, y=pair_grasps, pos=point_cloud,
                     pair_scores=pair_scores, sample_info=sample_info)
        if self.transform != None:
            data = self.transform(data)

        return data
    
    def grasps_contact_idxs(self, idx):
        i, j = self.triu_indices[0][idx], self.triu_indices[1][idx]
        return i, j
    
if __name__ == "__main__":

    import time
    from acronym_visualize_dataset import visualize_grasps
    train_data, valid_data = save_split_samples('../data', 5)
    dataset = TPPDataset(train_data)

    t = time.time()
    sample = dataset[1]
    print(sample)
    print(f"Time taken: {time.time() - t}")
    vertices = sample.pos.numpy()
    point_idx = np.random.randint(0, len(dataset.triu_indices[0]), 5)
    print(sample.pair_scores[:3])
    good_pair_scores_idxs = torch.where(sample.pair_scores > 0)[0][:3]
    print(good_pair_scores_idxs)
    grasps = sample.y[good_pair_scores_idxs, 0].numpy()
    contact_idx_arrays = dataset.grasps_contact_idxs(good_pair_scores_idxs)
    contact_idxs = [(p1, p2)  for p1 , p2 in zip(contact_idx_arrays[0], contact_idx_arrays[1])]
    
    print(contact_idxs)
    print(grasps)
    visualize_grasps(vertices, grasps, None, contact_idxs)


    

