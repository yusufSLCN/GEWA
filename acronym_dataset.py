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
        sample = self.data[idx]
        # Process the sample if needed
        model_file_name = sample['simplified_model_path']
        # mesh_scale = sample['scale']

        mesh_data = pywavefront.Wavefront(model_file_name)
        vertices = np.array(mesh_data.vertices)
        grasp_pose = sample['grasp_pose']
        success = sample['success']
        return vertices, grasp_pose, success
    
    def load_data(self, data_path):
        # Load the data from the specified path
        data = np.load(data_path, allow_pickle=True)
        return data

if __name__ == "__main__":
    dataset = AcronymDataset('train_simplified_acronym_samples.npy')
    print(len(dataset))
    print(dataset[0][1].shape)
    print(dataset[0][2])

