import torch
from torch.utils.data import Dataset
import numpy as np
import pywavefront
from transforms import create_random_rotation_translation_matrix
from create_dataset_paths import save_split_meshes
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
    
class AcronymDataset(Dataset):
    def __init__(self, data, crop_radius=None, transform=None, normalize_vertices=True):
        self.data = data
        self.transform = transform
        self.crop_radius = crop_radius
        self.normalize_vertices = normalize_vertices

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

        # Load the approach points
        approach_points_save_path = sample_info['approach_points_save_path']
        approach_point_counts = np.load(approach_points_save_path, allow_pickle=True)
        approach_point_counts = torch.tensor(approach_point_counts, dtype=torch.float32)

        # pick random point from the vertices
        # query_point_idx = np.random.randint(len(vertices))
        query_point_idx = 0
        query_point = vertices[query_point_idx].numpy().astype(np.float32)

        #normalize the vertices
        if self.normalize_vertices:
            mean = torch.mean(vertices, axis=0)
            vertices = vertices - mean
            sample_info['mean'] = mean.numpy().astype(np.float32)


        if self.crop_radius is not None and self.crop_radius > 0:
            # crop the vertices around the query point
            q_point = vertices[query_point_idx]
            crop_mask = np.linalg.norm(vertices - q_point, axis=1) < self.crop_radius
            vertices = vertices[crop_mask]
            query_point_idx = np.sum(crop_mask[:query_point_idx])
        
        # get the grasps of the point
        # some keys are not in the point grasp dict because of floating point errors
        query_point_key = tuple(np.round(query_point, 3))
        # while query_point_key not in point_grasp_dict:
        #     print(f"Query point {query_point_key} not in the point grasp dict. Picking a new point.")
        #     query_point_idx = np.random.randint(len(vertices))
        #     query_point = vertices[query_point_idx].numpy().astype(np.float32)
        #     query_point_key = tuple(np.round(query_point, 3))
        
        grasp_poses = point_grasp_dict[query_point_key]
        grasp_pose = grasp_poses[0][0]
        grasp_pose = torch.tensor(grasp_pose, dtype=torch.float32)
        grasp_pose_wo_aug = grasp_pose.clone()

        if self.normalize_vertices:
            grasp_pose[0:3, 3] = grasp_pose[0:3, 3] - mean
        
        sample_info['query_point'] = query_point
        sample_info['query_point_idx'] = query_point_idx

        grasp_pose = grasp_pose.view(-1)

        # success = torch.tensor(success)
        data = Data(x=vertices, y=grasp_pose, pos=vertices, query_point_idx=query_point_idx,
                     approach=approach_point_counts, sample_info=sample_info)
        if self.transform != None:
            data = self.transform(data)


        # calculate augmentation matrix
        aug_matrix =  data.y.view(4, 4) @ torch.inverse(grasp_pose_wo_aug) 
        data.sample_info["aug_matrix"] = aug_matrix
        return data
    
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

