import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from dataset.create_approach_dataset import save_split_samples
import numpy as np 

    
class ApproachDataset(Dataset):
    def __init__(self, data, transform=None, normalize_points=True, max_grasp_perpoint=20):
        self.data = data
        self.transform = transform
        self.normalize_points = normalize_points
        self.device = "cpu"
        self.max_grasp_perpoint = max_grasp_perpoint

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        # Process the sample if needed
        point_cloud = sample.point_cloud
        approach_scores = sample.approach_scores
        point_grasps = sample.point_grasps # grasp, contact1_idx, contact2_idx
        sample_info = sample.info
  
        # convert to torch tensors
        point_cloud = torch.tensor(point_cloud, dtype=torch.float32)
        approach_scores = torch.tensor(approach_scores, dtype=torch.float32)

        # contact_points = np.array([[point_grasps[i][1], point_grasps[i][2]]  for i in range(len(point_grasps))])
        # contact_points = torch.tensor(contact_points, dtype=torch.float32)
        num_grasps = torch.tensor([len(point_grasps[i]) for i in range(len(point_grasps))])
        num_grasps[num_grasps > self.max_grasp_perpoint] = self.max_grasp_perpoint
        grasps = np.zeros((len(point_grasps), self.max_grasp_perpoint, 4, 4))
        for i in range(len(point_grasps)):
            for j in range(self.max_grasp_perpoint):
                if j < len(point_grasps[i]):
                    grasps[i, j] = point_grasps[i][j]
                else:
                    break

        # grasps = np.array([point_grasps[i][0] for i in range(len(point_grasps))])
        point_grasps = torch.tensor(grasps, dtype=torch.float32)

        #normalize the vertices
        if self.normalize_points:
            mean = torch.mean(point_cloud, axis=0)
            point_cloud = point_cloud - mean
            sample_info['mean'] = mean
            point_grasps[:, :, 0:3, 3] = point_grasps[:, :, 0:3, 3] - mean

        # success = torch.tensor(success)
        data = Data(x=point_cloud, y=point_grasps, pos=point_cloud, num_grasps=num_grasps,
                     approach_scores=approach_scores, sample_info=sample_info)
        if self.transform != None:
            data = self.transform(data)

        return data
    
if __name__ == "__main__":
    train_data, valid_data = save_split_samples('../data', -1)
    dataset = ApproachDataset(train_data)
    print(dataset[0])
    # print(dataset[4207].num_grasps)
    bad_sample_count = 0
    for i in range(len(dataset)):   
        total_approach_scores = torch.sum(dataset[i].approach_scores)
        if total_approach_scores < 10:
            print(100 * '-')
            print(i)
            print(dataset[i])
            print(total_approach_scores)
            bad_sample_count += 1
    print(f"Bad samples: {bad_sample_count}")
    # print(dataset[0])

    # print(dataset[0][1].shape)
    # print(dataset[0][2])

