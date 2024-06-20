import torch
from torch.utils.data import Dataset
import numpy as np
import pywavefront
import torch_geometric as tg
from transforms import create_random_rotation_translation_matrix


class EdgegraspDataset(Dataset):
    def __init__(self, data_path, num_grasps=16, transform=None):
        self.data = self.load_data(data_path)
        self.transform = transform
        self.num_grasps = num_grasps
        
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
        grasp_indices = np.random.choice(len(vertices), size=self.num_grasps, replace=False)
        grasp_points = vertices[grasp_indices].numpy().astype(np.float32)
        grasp_poses = []
        for i, query_point in enumerate(grasp_points):
            # get the grasps of the point
            query_point_key = tuple(np.round(query_point, 3))
            while query_point_key not in point_grasp_dict:
                print(f"Query point {query_point_key} not in the point grasp dict. Picking a new point.")
                query_point_idx = np.random.randint(len(vertices))
                query_point = vertices[query_point_idx].numpy().astype(np.float32)
                query_point_key = tuple(np.round(query_point, 3))
                grasp_points[i] = query_point
                grasp_indices[i] = query_point_idx
            
            grasp_pose = point_grasp_dict[query_point_key][0][0]
            grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32)
            grasp_poses.append(grasp_pose) 
            
        grasp_poses = torch.stack(grasp_poses, dim=0)

        if self.transform != None:
            transform_matrix = create_random_rotation_translation_matrix(self.transform['rotation_range'], self.transform['translation_range'])
            transform_matrix = torch.tensor(transform_matrix, dtype=torch.float32)
            #add 1 to the vertices to make them homogeneous
            vertices = torch.cat((vertices, torch.ones((vertices.shape[0], 1))), 1)
            vertices = transform_matrix @ vertices.T
            vertices = vertices.T
            vertices = vertices[:, :3]

            grasp_poses = (transform_matrix @ grasp_poses.T).T

            grasp_points = vertices[grasp_indices]
        
        sample_info['grasp_points'] = grasp_points
        sample_info['grasp_indices'] = grasp_indices

        grasp_poses = grasp_poses.reshape((self.num_grasps, -1))
        # grasp_pose = sample_info['grasp_pose']
        # grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32)
        # grasp_pose = grasp_pose.view(-1)

        # success = torch.tensor(success)
        return vertices, grasp_poses, sample_info
    
    def load_data(self, data_path):
        # Load the data from the specified path
        data = np.load(data_path, allow_pickle=True)
        return data

if __name__ == "__main__":
    rotation_range = (-np.pi, np.pi)  # full circle range in radians
    translation_range = (-0.5, 0.5)  # translation values range
    transfom_params = {"rotation_range": (-np.pi, np.pi), "translation_range": (-0.5, 0.5)}
    dataset_transformed = EdgegraspDataset('sample_dirs/train_success_simplified_acronym_meshes.npy', transform=transfom_params)
    dataset = EdgegraspDataset('sample_dirs/train_success_simplified_acronym_meshes.npy')
    # print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2]["grasp_points"].shape)
    print(dataset[0][2]["grasp_indices"].shape)

    # print(dataset[0][1])
    # print(dataset_transformed[0][2])
    # print(dataset_transformed[0][1])
    
    # print(dataset[0][1].shape)
    # print(dataset[0][2])

