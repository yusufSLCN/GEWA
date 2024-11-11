import torch
from torch.utils.data import Dataset
import numpy as np
import pywavefront
from transforms import create_random_rotation_translation_matrix
from deprecated.create_dataset_paths import save_split_meshes
from torch_geometric.data import Data
import trimesh


class RandomRotationTransform:
    def __init__(self, rotation_range):
        self.rotation_range = rotation_range

    def __call__(self, data):
        vertices = data.pos.numpy().astype(np.float32)
        grasp = data.y.numpy().astype(np.float32).reshape(4, 4)
        
        # Apply random rotation to vertices
        rotation_angle = np.random.uniform(*self.rotation_range)
        rotation_matrix = trimesh.transformations.rotation_matrix(rotation_angle, [0, 0, 1])
        rotated_vertices = np.dot(rotation_matrix[:3, :3], vertices.T).T

        # Apply random rotation to ground truth grasp
        rotated_grasp = np.dot(rotation_matrix, grasp)

        # Update the data object with the rotated vertices and grasp
        data.pos = torch.from_numpy(rotated_vertices).float()
        data.y = torch.from_numpy(rotated_grasp).flatten().float()

        return data
    
class EdgegraspDataset(Dataset):
    def __init__(self, data, num_grasps=16, transform=None):
        self.data = data
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
        
        
        grasp_pose = grasp_poses[0][0]
        grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32)
        # grasp_pose[0:3, 3] = grasp_pose[0:3, 3] - mean_pos
        
        sample_info['query_point'] = query_point
        sample_info['query_point_idx'] = query_point_idx

        grasp_poses = grasp_poses.reshape((self.num_grasps, -1))

        # success = torch.tensor(success)
        data = Data(x=vertices, y=grasp_pose, pos=vertices, sample_info=sample_info)
        if self.transform != None:
            data = self.transform(data)
            # sample_info["query_point"] = data.pos[data.sample_info['query_point_idx']]
            # data.sample_info = sample_info

        return data
    
if __name__ == "__main__":
    train_data, valid_data = save_split_meshes('../data', 100)
    dataset_transformed = EdgegraspDataset(train_data)
    dataset = EdgegraspDataset(valid_data)
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

