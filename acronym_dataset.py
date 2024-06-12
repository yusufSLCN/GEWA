import torch
from torch.utils.data import Dataset
import numpy as np
import pywavefront


class AcronymDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_info = self.data[idx]
        # Process the sample if needed
        model_file_name = sample_info['simplified_model_path']
        mesh_data = pywavefront.Wavefront(model_file_name)
        vertices = np.array(mesh_data.vertices)
        vertices = torch.tensor(vertices, dtype=torch.float32)
        grasp_pose = sample_info['grasp_pose']
        grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32)
        grasp_pose = grasp_pose.view(-1)
        # success = sample['success']
        mesh_scale = float(sample_info['scale'])
        vertices = vertices * mesh_scale

        # success = torch.tensor(success)
        return vertices, grasp_pose, sample_info
    
    def load_data(self, data_path):
        # Load the data from the specified path
        data = np.load(data_path, allow_pickle=True)
        return data

if __name__ == "__main__":
    dataset = AcronymDataset('train_success_simplified_acronym_samples.npy')
    print(len(dataset))
    print(dataset[0][1].shape)
    print(dataset[0][2])

