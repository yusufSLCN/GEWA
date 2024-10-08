import os
import numpy as np
from Sample import Sample
import tqdm
from acronym_utils import load_file_names, extract_sample_info, find_appraoch_score_target, create_point_close_grasp_list_and_approach_scores
import h5py
import open3d as o3d


def save_split_samples(data_dir, num_mesh, train_ratio=0.8):
    if not os.path.exists('sample_dirs'):
        os.makedirs('sample_dirs')
    paths_dir = os.path.join('sample_dirs', f'gewa_r_{num_mesh}_samples.npy')
    if os.path.exists(paths_dir):
        samples = np.load(paths_dir, allow_pickle=True)

    else:
        samples = get_point_cloud_samples(data_dir, num_mesh=num_mesh, min_num_grasps=100)
        #sort by name 
        samples = sorted(samples, key=lambda x: x.simplified_mesh_path)
        np.random.seed(0)
        np.random.shuffle(samples)
        np.save(paths_dir, samples)

    #split samples into train and test sets
    if len(samples) <= num_mesh or num_mesh < 0:
        num_mesh = len(samples)
        print(f"Number of meshes in the simlified subset: {num_mesh}")

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


def get_point_cloud_samples(data_dir, success_threshold=0.5, num_mesh=-1, num_points=1000, min_num_grasps=100):
    simplified_mesh_directory = os.path.join(data_dir, 'simplified_obj')
    grasp_directory =  os.path.join(data_dir, 'acronym/grasps')
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'
    point_grasps_folder = os.path.join(data_dir, f'point_grasps_{num_points}')
    approach_points_folder = os.path.join(data_dir, f'approach_points_{num_points}')
    point_cloud_folder = os.path.join(data_dir, f'point_cloud_{num_points}')

    if not os.path.exists(approach_points_folder):
        print(f"Creating directory {approach_points_folder}")
        os.makedirs(approach_points_folder)

    if not os.path.exists(point_grasps_folder):
        print(f"Creating directory {point_grasps_folder}")
        os.makedirs(point_grasps_folder)

    if not os.path.exists(point_cloud_folder):
        print(f"Creating directory {point_cloud_folder}")
        os.makedirs(point_cloud_folder)

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
        approach_points_save_path = f'{approach_points_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        point_grasps_save_path = f'{point_grasps_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        point_cloud_save_path = f'{point_cloud_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        # Check if the simplified mesh exists because not all samples have been simplified
        if os.path.exists(simplified_mesh_path):
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

                if not (os.path.exists(point_cloud_save_path) and os.path.exists(approach_points_save_path) and os.path.exists(point_grasps_save_path)):
                    mesh_data = o3d.io.read_triangle_mesh(simplified_mesh_path)
                    point_cloud = mesh_data.sample_points_poisson_disk(num_points)
                    sample['scale'] = float(sample['scale'])
                    point_cloud = np.asarray(point_cloud.points) * sample['scale']

                    point_grasp_list, approach_point_scores = create_point_close_grasp_list_and_approach_scores(point_cloud, success_grasp_poses, radius=0.01)
                    if np.sum(approach_point_scores) < min_num_grasps:
                        continue
                    point_grasp_array = np.array(point_grasp_list, dtype=object)
                    np.save(point_grasps_save_path, point_grasp_array)
                    np.save(approach_points_save_path, approach_point_scores)

                    np.save(point_cloud_save_path, point_cloud)

                simplified_mesh_count += 1
                point_cloud_sample = Sample(simplified_mesh_path, point_cloud_save_path, approach_points_save_path, point_grasps_save_path, grasps_file_name, sample)
                point_cloud_samples.append(point_cloud_sample)
    return point_cloud_samples

if __name__ == "__main__":
    train_samples, val_samples = save_split_samples('../data', -1)
    print(f"Number of train samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(val_samples)}")
    print("Done!")

    samp = train_samples[0]
    print(samp.info)