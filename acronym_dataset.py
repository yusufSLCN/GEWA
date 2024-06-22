import torch
from torch.utils.data import Dataset
import numpy as np
import pywavefront
from transforms import create_random_rotation_translation_matrix
from create_dataset_paths import save_split_meshes
from torch_geometric.data import Data

class AcronymDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample_info = self.data[idx]
        # Process the sample if needed
        model_file_name = sample_info['simplified_model_path']
        mesh_data = pywavefront.Wavefront(model_file_name)
        vertices = np.array(mesh_data.vertices)

        vertices = torch.tensor(vertices, dtype=torch.float32)
  
        # success = sample['success']
        mesh_scale = float(sample_info['scale'])
        vertices = vertices * mesh_scale

        # Load the point grasp data
        point_grasp_save_path = sample_info['point_grasp_save_path']
        point_grasp_dict = np.load(point_grasp_save_path, allow_pickle=True).item()

        # pick random point from the vertices
        # query_point_idx = np.random.randint(len(vertices))
        query_point_idx = 0
        query_point = vertices[query_point_idx].numpy().astype(np.float32)
        
        # get the grasps of the point
        query_point_key = tuple(np.round(query_point, 3))
        while query_point_key not in point_grasp_dict:
            print(f"Query point {query_point_key} not in the point grasp dict. Picking a new point.")
            query_point_idx = np.random.randint(len(vertices))
            query_point = vertices[query_point_idx].numpy().astype(np.float32)
            query_point_key = tuple(np.round(query_point, 3))
        
        grasp_poses = point_grasp_dict[query_point_key]
        
        # not select a random grasp
        # grasp_idx = np.random.randint(len(grasp_poses))
        # grasp_pose = grasp_poses[grasp_idx][0]
        grasp_pose = grasp_poses[0][0]
        grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32)

        # if self.transform != None:
        #     transform_matrix = create_random_rotation_translation_matrix(self.transform['rotation_range'], self.transform['translation_range'])
        #     transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32)
        #     #add 1 to the vertices to make them homogeneous
        #     vertices = torch.cat((vertices, torch.ones((vertices.shape[0], 1))), 1)
        #     vertices = transform_matrix @ vertices.T
        #     vertices = vertices.T
        #     vertices = vertices[:, :3]

        #     grasp_pose = transform_matrix @ grasp_pose 

        #     query_point = vertices[query_point_idx]
        
        sample_info['query_point'] = query_point
        sample_info['query_point_idx'] = query_point_idx

        grasp_pose = grasp_pose.view(-1)
        # grasp_pose = sample_info['grasp_pose']
        # grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32)
        # grasp_pose = grasp_pose.view(-1)

        # success = torch.tensor(success)
        data = Data(x=vertices, y=grasp_pose, pos=vertices, sample_info=sample_info)
        if self.transform != None:
            data = self.transform(data)
            # sample_info["query_point"] = data.pos[data.sample_info['query_point_idx']]
            # data.sample_info = sample_info

        return data
    
    # def load_data(self, data_path):
    #     # Load the data from the specified path
    #     data = np.load(data_path, allow_pickle=True)
    #     return data

if __name__ == "__main__":

    train_data, valid_data = save_split_meshes('../data', 100)
    dataset_transformed = AcronymDataset(train_data)
    dataset = AcronymDataset(valid_data)
    print(len(dataset))
    print(dataset[0][2])
    print(dataset[0][1])
    print(dataset_transformed[0][2])
    print(dataset_transformed[0][1])
    
    # print(dataset[0][1].shape)
    # print(dataset[0][2])

