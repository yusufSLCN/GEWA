import torch
from torch.utils.data import Dataset
import numpy as np
from acronym_utils import load_sample, convert2graph

class AcronymDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_path = self.data[idx]
        # Process the sample if needed
        T, success, vertices = load_sample(sample_path)
        # vertices, A = convert2graph(vertices)
        return vertices, T, success
    
    def load_data(self, data_path):
        # Load the data from the specified path
        data = np.load(data_path, allow_pickle=True)
        return data

if __name__ == "__main__":
    dataset = AcronymDataset('simplified_acronym_samples.npy')
    print(len(dataset))
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)

