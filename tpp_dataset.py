import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from create_tpp_dataset import save_split_samples
import numpy as np 
import time
    
class TPPDataset(Dataset):
    def __init__(self, data, transform=None, return_pair_matrix=False):
        self.data = data
        self.transform = transform
        self.device = "cpu"
        self.triu_indices = np.triu_indices(1000, k=1)
        self.return_pair_matrix = return_pair_matrix


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Process the sample if needed
        point_cloud = sample.point_cloud
        pair_scores = sample.pair_scores
        pair_grasps_dict = sample.pair_grasps 
        sample_info = sample.info

        if self.return_pair_matrix:
            pair_score_matrix = sample.pair_score_matrix
            pair_score_matrix = torch.tensor(pair_score_matrix, dtype=torch.float32)
        else:
            pair_score_matrix = None
  
        # convert to torch tensors
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        pair_scores = torch.tensor(pair_scores, dtype=torch.float32)

        data = Data(x=point_cloud, y=pair_grasps_dict, pos=point_cloud,
                     pair_scores=pair_scores, pair_score_matrix=pair_score_matrix, sample_info=sample_info)
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


    

