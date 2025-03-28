import os
import numpy as np
from dataset.TPPSample import TPPSample
import tqdm
from utils.acronym_utils import load_file_names, extract_sample_info, extract_sample_info_from_basename
import h5py
import json
import torch

def save_split_samples(data_dir, num_mesh, dataset_name="tpp_effdict", radius=0.005, train_ratio=0.8, contactnet_split=False):
    if not os.path.exists('sample_dirs'):
        os.makedirs('sample_dirs')
    paths_dir = os.path.join('sample_dirs', f'{dataset_name}_r-{radius}_{num_mesh}_samples.npy')
    print(paths_dir)
    if os.path.exists(paths_dir):
        samples = np.load(paths_dir, allow_pickle=True)
    else:
        samples = get_point_cloud_samples(data_dir, dataset_name=dataset_name, num_mesh=num_mesh, min_num_grasps=100, radius=radius)
        #sort by name 
        samples = sorted(samples, key=lambda x: x.simplified_mesh_path)
        np.random.seed(0)
        np.random.shuffle(samples)
        np.save(paths_dir, samples)

    #split samples into train and test sets
    if len(samples) <= num_mesh or num_mesh < 0:
        num_mesh = len(samples)
        print(f"Number of meshes in the simlified subset: {num_mesh}")

    if contactnet_split:
        print("Using contactnet split")
        train_meshes, valid_meshes = get_contactnet_split()
        train_samples = []
        valid_samples = []
        for sample in samples:
            mesh_name = sample.info['grasps']
            mesh_name = os.path.basename(mesh_name)
            mesh_name = os.path.splitext(mesh_name)[0]
            if mesh_name in train_meshes:
                train_samples.append(sample)
            elif mesh_name in valid_meshes:
                valid_samples.append(sample)
    else:
        split_idx = int(len(samples) * train_ratio)
        all_train_samples = samples[:split_idx]
        all_valid_samples = samples[split_idx:]

        subset_idx = int(num_mesh * train_ratio)
        train_samples = all_train_samples[:subset_idx]
        valid_samples = all_valid_samples[:num_mesh - subset_idx]

    # #save the train and test meshes
    # np.save('sample_dirs/train_success_simplified_acronym_meshes.npy', train_meshes)
    # np.save('sample_dirs/valid_success_simplified_acronym_meshes.npy', valid_meshes)
    print(f"Train mesh {len(train_samples)}")
    print(f"Test mesh {len(valid_samples)}")
    return train_samples, valid_samples

def save_contactnet_split_samples(data_dir, num_mesh, dataset_name="tpp_effdict", radius=0.005):
    from dataset.analyze_contactnet_split import create_contactnet_splits
    if not os.path.exists('sample_dirs'):
        os.makedirs('sample_dirs')
    
    dataset_name = f"{dataset_name}_csplit"
    paths_dir = os.path.join('sample_dirs', f'{dataset_name}_r-{radius}_{num_mesh}_samples.npy')
    print(paths_dir)
    if os.path.exists(paths_dir):
        samples = np.load(paths_dir, allow_pickle=True).item()
        train_samples = samples['train']
        valid_samples = samples['valid']
    else:
        train_meshes, valid_meshes = create_contactnet_splits(num_mesh=num_mesh, min_num_success_grasps=100)

        samples = {}
        train_samples = process_contactnet_split(train_meshes, data_dir, dataset_name, success_threshold=0.5, num_points=1000, radius=0.005)
        samples['train'] = train_samples
        valid_samples = process_contactnet_split(valid_meshes, data_dir, dataset_name, success_threshold=0.5, num_points=1000, radius=0.005)
        samples['valid'] = valid_samples
        np.save(paths_dir, samples)

    return train_samples, valid_samples

def process_contactnet_split(split_file, data_dir, dataset_name, success_threshold=0.5, num_points=1000, radius=0.005):
    import open3d as o3d

    if os.path.exists(split_file):
        mesh_names = np.load(split_file)
    else:
        print(f"Split file not found: {split_file}")
        return
    
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'

    simplified_mesh_directory = os.path.join(data_dir, 'simplified_obj')
    grasp_directory =  os.path.join(data_dir, 'acronym/grasps')
    sample_info = extract_sample_info_from_basename(mesh_names, model_root=model_root, grasp_root=grasp_directory)

   
    pair_grasp_folder = os.path.join(data_dir, f'{dataset_name}_grasps_{num_points}_{radius}')
    point_cloud_folder = os.path.join(data_dir, f'{dataset_name}_point_cloud_{num_points}_{radius}')
    touch_pair_score_folder = os.path.join(data_dir, f'{dataset_name}_scores_{num_points}_{radius}')
    touch_pair_score_matrix_folder = os.path.join(data_dir, f'{dataset_name}_score_matrix_{num_points}_{radius}')
    normals_folder = os.path.join(data_dir, f'{dataset_name}_normals_{num_points}_{radius}')

    if not os.path.exists(touch_pair_score_matrix_folder):
        print(f"Creating directory {touch_pair_score_matrix_folder}")
        os.makedirs(touch_pair_score_matrix_folder)

    if not os.path.exists(touch_pair_score_folder):
        print(f"Creating directory {touch_pair_score_folder}")
        os.makedirs(touch_pair_score_folder)

    if not os.path.exists(point_cloud_folder):
        print(f"Creating directory {point_cloud_folder}")
        os.makedirs(point_cloud_folder)

    if not os.path.exists(pair_grasp_folder):
        print(f"Creating directory {pair_grasp_folder}")
        os.makedirs(pair_grasp_folder)

    if not os.path.exists(normals_folder):
        print(f"Creating directory {normals_folder}")
        os.makedirs(normals_folder)


    point_cloud_samples = []
    # Dictionary to store the closese grasps to each point in the mesh
    print("Extracting simplified meshes with closest grasps")
    for i, sample in tqdm.tqdm(enumerate(sample_info), total=len(sample_info)):
        simplified_mesh_path = f'{simplified_mesh_directory}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.obj'
        pair_scores_path = f'{touch_pair_score_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_score_matrix_path = f'{touch_pair_score_matrix_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        point_cloud_path = f'{point_cloud_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_grasps_path = f'{pair_grasp_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        normals_path = f'{normals_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'

        # Check if the simplified mesh exists because not all samples have been simplified
        if os.path.exists(simplified_mesh_path):
            grasps_file_name = sample['grasps']
            data = h5py.File(grasps_file_name, "r")
            grasp_poses = np.array(data["grasps/transforms"])
            grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])

            success_grasp_mask = grasp_success > success_threshold
            success_grasp_poses = grasp_poses[success_grasp_mask]
            sample['scale'] = float(sample['scale'])
            if not (os.path.exists(point_cloud_path) and os.path.exists(pair_scores_path) and os.path.exists(pair_grasps_path) and os.path.exists(pair_score_matrix_path)):
                mesh_data = o3d.io.read_triangle_mesh(simplified_mesh_path)
                o3d.utility.random.seed(0)
                point_cloud_data = mesh_data.sample_points_poisson_disk(num_points)
                
                point_cloud = np.asarray(point_cloud_data.points) * sample['scale']

                point_cloud_data.estimate_normals()
                normals = np.asarray(point_cloud_data.normals)

                #normalize point cloud
                # mean = np.mean(point_cloud, axis=0)
                # sample['mean'] = mean
                # point_cloud = point_cloud - mean
                # success_grasp_poses[:, :3, 3] = success_grasp_poses[:, :3, 3] - mean

                pair_score_matrix, pair_scores, tpp_grasp_dict, _ = create_touch_point_pair_scores_and_grasps(point_cloud, success_grasp_poses, cylinder_radius=radius, cylinder_height=0.041)
                num_positive_pairs = torch.sum(pair_scores)
                if  num_positive_pairs < 50:
                    print(f"Skipping {simplified_mesh_path}, pos pair count = {num_positive_pairs}")
                    continue

                np.save(pair_score_matrix_path, pair_score_matrix)
                np.save(pair_scores_path, pair_scores)
                np.save(pair_grasps_path, tpp_grasp_dict)
                np.save(point_cloud_path, point_cloud)
                np.save(normals_path, normals)
                
            point_cloud_sample = TPPSample(simplified_mesh_path, point_cloud_path, pair_scores_path, pair_score_matrix_path,
                                            pair_grasps_path, normals_path, grasps_file_name, sample)
            point_cloud_samples.append(point_cloud_sample)
    return point_cloud_samples

    



def get_point_cloud_samples(data_dir, dataset_name, success_threshold=0.5, num_mesh=-1, num_points=1000, min_num_grasps=100, radius=0.005):
    import open3d as o3d
   
    simplified_mesh_directory = os.path.join(data_dir, 'simplified_obj')
    grasp_directory =  os.path.join(data_dir, 'acronym/grasps')
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'
    pair_grasp_folder = os.path.join(data_dir, f'{dataset_name}_grasps_{num_points}_{radius}')
    point_cloud_folder = os.path.join(data_dir, f'{dataset_name}_point_cloud_{num_points}_{radius}')
    touch_pair_score_folder = os.path.join(data_dir, f'{dataset_name}_scores_{num_points}_{radius}')
    touch_pair_score_matrix_folder = os.path.join(data_dir, f'{dataset_name}_score_matrix_{num_points}_{radius}')
    normals_folder = os.path.join(data_dir, f'{dataset_name}_normals_{num_points}_{radius}')

    if not os.path.exists(touch_pair_score_matrix_folder):
        print(f"Creating directory {touch_pair_score_matrix_folder}")
        os.makedirs(touch_pair_score_matrix_folder)

    if not os.path.exists(touch_pair_score_folder):
        print(f"Creating directory {touch_pair_score_folder}")
        os.makedirs(touch_pair_score_folder)

    if not os.path.exists(point_cloud_folder):
        print(f"Creating directory {point_cloud_folder}")
        os.makedirs(point_cloud_folder)

    if not os.path.exists(pair_grasp_folder):
        print(f"Creating directory {pair_grasp_folder}")
        os.makedirs(pair_grasp_folder)

    if not os.path.exists(normals_folder):
        print(f"Creating directory {normals_folder}")
        os.makedirs(normals_folder)
    
    grasp_file_names = load_file_names(grasp_directory)
    # read discarded_samples.txt
    discarded_samples = []
    if os.path.exists('discarded_objects.txt'):
        with open('discarded_objects.txt', 'r') as f:
            discarded_samples = f.readlines()
            discarded_samples = [sample.strip() for sample in discarded_samples]

    sample_info = extract_sample_info(grasp_file_names, model_root=model_root, discard_samples=discarded_samples)
    point_cloud_samples = []
    # Dictionary to store the closese grasps to each point in the mesh
    simplified_mesh_count = 0
    print("Extracting simplified meshes with closest grasps")
    for i, sample in tqdm.tqdm(enumerate(sample_info), total=len(sample_info)):
        simplified_mesh_path = f'{simplified_mesh_directory}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.obj'
        pair_scores_path = f'{touch_pair_score_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_score_matrix_path = f'{touch_pair_score_matrix_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        point_cloud_path = f'{point_cloud_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_grasps_path = f'{pair_grasp_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        normals_path = f'{normals_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'

        # Check if the simplified mesh exists because not all samples have been simplified
        if os.path.exists(simplified_mesh_path):
            # print(f"Processing {simplified_mesh_count}/{num_mesh}")
            if num_mesh > 0 and simplified_mesh_count >= num_mesh:
                return point_cloud_samples
            
            grasps_file_name = sample['grasps']
            data = h5py.File(grasps_file_name, "r")
            grasp_poses = np.array(data["grasps/transforms"])
            grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
            num_success = np.sum(grasp_success > success_threshold)
            if num_success > min_num_grasps:
                success_grasp_mask = grasp_success > success_threshold
                success_grasp_poses = grasp_poses[success_grasp_mask]

                if not (os.path.exists(point_cloud_path) and os.path.exists(pair_scores_path) and os.path.exists(pair_grasps_path) and os.path.exists(pair_score_matrix_path)):
                    mesh_data = o3d.io.read_triangle_mesh(simplified_mesh_path)
                    o3d.utility.random.seed(0)
                    point_cloud_data = mesh_data.sample_points_poisson_disk(num_points)
                    sample['scale'] = float(sample['scale'])
                    point_cloud = np.asarray(point_cloud_data.points) * sample['scale']

                    point_cloud_data.estimate_normals()
                    normals = np.asarray(point_cloud_data.normals)

                    #normalize point cloud
                    # mean = np.mean(point_cloud, axis=0)
                    # sample['mean'] = mean
                    # point_cloud = point_cloud - mean
                    # success_grasp_poses[:, :3, 3] = success_grasp_poses[:, :3, 3] - mean

                    pair_score_matrix, pair_scores, tpp_grasp_dict, _ = create_touch_point_pair_scores_and_grasps(point_cloud, success_grasp_poses, cylinder_radius=radius, cylinder_height=0.041)
                    if np.sum(pair_score_matrix) < min_num_grasps:
                        continue
                    np.save(pair_score_matrix_path, pair_score_matrix)
                    np.save(pair_scores_path, pair_scores)
                    np.save(pair_grasps_path, tpp_grasp_dict)
                    np.save(point_cloud_path, point_cloud)
                    np.save(normals_path, normals)
                # else:
                #     point_cloud = np.load(point_cloud_path)
                #     mean = np.mean(point_cloud, axis=0)
                #     sample['mean'] = mean
                #     sample['scale'] = float(sample['scale'])
                
                point_cloud_sample = TPPSample(simplified_mesh_path, point_cloud_path, pair_scores_path, pair_score_matrix_path,
                                               pair_grasps_path, normals_path, grasps_file_name, sample)
                simplified_mesh_count += 1
                point_cloud_samples.append(point_cloud_sample)
    return point_cloud_samples


def create_touch_point_pair_scores_and_grasps(vertices, grasp_poses, cylinder_radius=0.02, cylinder_height=0.02, max_grasps_per_pair=20, pair_alignment_threshold=0.7):
    import torch
    
    gripper_right_tip_vector = np.array([-4.100000e-02, -7.27595772e-12, 1.12169998e-01, 1])
    gripper_left_tip_vector = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01, 1])
    right_tip_pos = np.matmul(grasp_poses, gripper_right_tip_vector)[:, :3]
    left_tip_pos = np.matmul(grasp_poses, gripper_left_tip_vector)[:, :3]
    tip_axis =  right_tip_pos - left_tip_pos
    tip_axis = tip_axis / np.linalg.norm(tip_axis, axis=1).reshape(-1, 1)
    left_cylinder_bottom = left_tip_pos + tip_axis * cylinder_height
    right_cylinder_bottom = right_tip_pos - tip_axis * cylinder_height

    point_pair_score_matrix = np.zeros((vertices.shape[0], vertices.shape[0]), dtype=int)
    upper_tri_idx = np.triu_indices(vertices.shape[0], k=1)
    # tpp_grasps_matrix = np.zeros((vertices.shape[0], vertices.shape[0], max_grasps_per_pair, 4, 4))
    # grasp_counts = np.zeros((len(upper_tri_idx[0])))

    tpp_grasp_dict = {}
    # for (i, j) in zip(upper_tri_idx[0], upper_tri_idx[1]):
    #     tpp_grasp_dict[frozenset((i, j))] = []

    # print(left_tip_pos.shape, left_cylinder_bottom.shape, right_tip_pos.shape, right_cylinder_bottom.shape, grasp_poses.shape)
    for (left_tip, left_bottom, right_tip, right_bottom, pose) in zip(left_tip_pos, left_cylinder_bottom, right_tip_pos, right_cylinder_bottom,  grasp_poses):
        # print(f"Left tip: {left_tip}, Left bottom: {left_bottom}, Right tip: {right_tip}, Right bottom: {right_bottom}")
        inside_right_cylinder = is_point_inside_cylinder(vertices, right_tip, right_bottom, cylinder_radius)
        inside_left_cylinder = is_point_inside_cylinder(vertices, left_tip, left_bottom, cylinder_radius)

        # print(inside_right_cylinder.shape, inside_left_cylinder.shape)
        right_tip_touch_points_idxs = inside_right_cylinder.nonzero()[0]
        left_tip_touch_points_idxs = inside_left_cylinder.nonzero()[0]

        cylinder_axis = left_tip - right_tip
        cylinder_axis = cylinder_axis / np.linalg.norm(cylinder_axis)

        for i in left_tip_touch_points_idxs:
            for j in right_tip_touch_points_idxs:
                if i == j:
                    continue
                if point_pair_score_matrix[i, j] >= max_grasps_per_pair:
                    continue
                

                pair_vector = vertices[j] - vertices[i]
                pair_vector = pair_vector / np.linalg.norm(pair_vector)
                dot_product = np.abs(np.dot(cylinder_axis, pair_vector))
                # check if any of the larger than 0.7
                if dot_product < pair_alignment_threshold:
                    continue

                point_pair_score_matrix[i, j] += 1
                # point_pair_score_matrix[j, i] += 1
                pair_key = frozenset((i, j))
                if pair_key in tpp_grasp_dict:
                    tpp_grasp_dict[pair_key].append(pose)
                else:
                    tpp_grasp_dict[pair_key] = [pose]

    #convert dictionary lists to np arrays
    for key in tpp_grasp_dict:
        tpp_grasp_dict[key] = np.array(tpp_grasp_dict[key])

    pair_scores = torch.zeros((len(upper_tri_idx[0])), dtype=torch.int16)
    for i in range(pair_scores.shape[0]):
        num_pair_grasps = 0
        num_pair_grasps = point_pair_score_matrix[upper_tri_idx[0][i], upper_tri_idx[1][i]]
        num_pair_grasps += point_pair_score_matrix[upper_tri_idx[1][i], upper_tri_idx[0][i]]
        pair_scores[i] = num_pair_grasps

    # print(point_pair_score_matrix)
    # pair_scores = point_pair_score_matrix[upper_tri_idx]
    # tpp_grasps = tpp_grasps_matrix[upper_tri_idx]
    cylinder_edges = (right_tip_pos, right_cylinder_bottom, left_tip_pos, left_cylinder_bottom)
    return point_pair_score_matrix, pair_scores, tpp_grasp_dict, cylinder_edges

        

def is_point_inside_cylinder(points, cylinder_top, cylinder_bottom, cylinder_radius):
    # Calculate the vector from the bottom point to the top point
    cylinder_axis = cylinder_top - cylinder_bottom
    
    # Calculate the vector from the bottom point to the given points
    point_vectors = points - cylinder_bottom.reshape(1, 3)
    
    # Calculate the projection of the point vectors onto the cylinder axis
    projections = np.dot(point_vectors, cylinder_axis) / np.dot(cylinder_axis, cylinder_axis)
    
    # Calculate the closest points on the cylinder axis to the given points
    closest_points = cylinder_bottom + projections.reshape(-1, 1) * cylinder_axis
    # Calculate the distances between the given points and the closest points on the cylinder axis
    distances = np.linalg.norm(points - closest_points, axis=1)
    
    # Check if the distances are within the radius of the cylinder
    within_radius = distances <= cylinder_radius
    
    # Check if the projections are within the height of the cylinder
    within_height = (projections >= 0) & (projections <= 1)
    
    # Check if all points are inside the cylinder
    inside_cylinder = np.logical_and(within_radius, within_height)
    
    return inside_cylinder



def create_point_cloud_and_grasps(N, num_grasps):
    import open3d as o3d

    gripper_right_tip_vector = np.array([-5.100000e-02, 0, 1.12169998e-01, 1])
    gripper_left_tip_vector = np.array([5.10000000e-02, 0, 1.12169998e-01, 1])
    cube_size = 0.08
    mid_point = (gripper_right_tip_vector[:3] + gripper_left_tip_vector[:3]) / 2
    points = np.random.uniform(-cube_size/2, cube_size/2, size=(N, 3)) + mid_point
    dummy_grasp_poses = np.eye(4).reshape(1, 4, 4).repeat(num_grasps, axis=0)
    for i in range(1, num_grasps):
        dummy_grasp_poses[i, :3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, i * np.pi/num_grasps])
    return points, dummy_grasp_poses


def get_contactnet_split(num_mesh=-1):
    path = 'splits'
    train_meshes = []
    valid_meshes = []
    split_paths = os.listdir(path)
    split_paths = sorted(split_paths)
    #json files in the path
    for model_split_file in split_paths:
        if num_mesh > 0 and len(train_meshes) >= num_mesh:
            break
        if model_split_file.endswith(".json"):
            with open(os.path.join(path, model_split_file), 'r') as file:
                data = json.load(file)

                train_names = [os.path.splitext(file_name)[0] for file_name in data['train']]
                test_names = [os.path.splitext(file_name)[0] for file_name in data['test']]
                # print(train_names)
                # print(test_names)
                train_meshes += train_names
                valid_meshes += test_names
    print("contactnet split")
    print(f"Train split {len(train_meshes)}")
    print(f"Test split {len(valid_meshes)}")
    return train_meshes, valid_meshes

if __name__ == "__main__":
    
    # train_samples, val_samples = save_split_samples('../data',  400, dataset_name="tpp_effdict_nomean_wnormals", contactnet_split=True)
    
    train_samples, val_samples = save_contactnet_split_samples('../data',  5600, dataset_name="tpp_effdict_nomean_wnormals")
    # train_samples, val_samples = get_contactnet_split()
    print(f"Number of train samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(val_samples)}")
    print("Done!")
    