import os
import open3d as o3d
import numpy as np
import h5py
from acronym_utils import load_file_names, extract_sample_info
import tqdm
from TPPSample import TPPSample
from create_tpp_dataset import create_touch_point_pair_scores_and_grasps
from visualize_tpp_dataset import show_pair_edges
from acronym_visualize_dataset import visualize_grasp

def get_point_cloud_samples(data_dir, success_threshold=0.5, num_mesh=-1, num_points=1000, min_num_grasps=100, radius=0.005):
    grasp_directory =  os.path.join(data_dir, 'acronym/grasps')
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'
    pair_grasp_folder = os.path.join(data_dir, f'tpp_origin_grasps_{num_points}_{radius}')
    point_cloud_folder = os.path.join(data_dir, f'tpp_origin_point_cloud_{num_points}_{radius}')
    touch_pair_score_folder = os.path.join(data_dir, f'tpp_origin_scores_{num_points}_{radius}')
    touch_pair_score_matrix_folder = os.path.join(data_dir, f'tpp_origin_score_matrix_{num_points}_{radius}')

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

    grasp_file_names = load_file_names(grasp_directory)
    # read discarded_samples.txt
    discarded_samples = []
    # if os.path.exists('discarded_objects.txt'):
    #     with open('discarded_objects.txt', 'r') as f:
    #         discarded_samples = f.readlines()
    #         discarded_samples = [sample.strip() for sample in discarded_samples]

    sample_info = extract_sample_info(grasp_file_names, model_root=model_root, discard_samples=discarded_samples)
    point_cloud_samples = []
    # Dictionary to store the closese grasps to each point in the mesh
    processed_mesh_count = 0

    for i, sample in tqdm.tqdm(enumerate(sample_info), total=len(sample_info)):
        pair_scores_path = f'{touch_pair_score_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_score_matrix_path = f'{touch_pair_score_matrix_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        point_cloud_path = f'{point_cloud_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_grasps_path = f'{pair_grasp_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        model_path = f'{model_root}/{sample["model_name"]}.obj'

        grasp_file_path = sample['grasps']
        if not (os.path.exists(model_path) and os.path.exists(grasp_file_path)):
            continue
        print(f"Processing {model_path}, {grasp_file_path}")

        if num_mesh > 0 and processed_mesh_count >= num_mesh:
            return point_cloud_samples
        
        data = h5py.File(grasp_file_path, "r")
        grasp_poses = np.array(data["grasps/transforms"])
        grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        num_success = np.sum(grasp_success > success_threshold)
        if num_success < min_num_grasps:
            print(f"Skipping {model_path} because it has {num_success} successful grasps")
            continue

        success_grasp_mask = grasp_success > success_threshold
        success_grasp_poses = grasp_poses[success_grasp_mask]
        if not (os.path.exists(point_cloud_path) and os.path.exists(pair_scores_path) and os.path.exists(pair_grasps_path) and os.path.exists(pair_score_matrix_path)):
            mesh_data = o3d.io.read_triangle_mesh(model_path)
            sample['scale'] = float(sample['scale'])
            # mesh_data.scale(sample['scale'], center=mesh_data.get_center())
            o3d.utility.random.seed(0)
            point_cloud = mesh_data.sample_points_poisson_disk(num_points)
            # o3d.visualization.draw_geometries([mesh_data, point_cloud])
            point_cloud = np.asarray(point_cloud.points) * sample['scale']

            #normalize point cloud
            mean = np.mean(point_cloud, axis=0)
            point_cloud = point_cloud - mean
            success_grasp_poses[:, :3, 3] = success_grasp_poses[:, :3, 3] - mean
            visualize_grasp(point_cloud, success_grasp_poses[0])

            pair_score_matrix, pair_scores, tpp_grasp_dict, _ = create_touch_point_pair_scores_and_grasps(point_cloud, success_grasp_poses, cylinder_radius=radius, cylinder_height=0.041)
            num_grasp_pairs = np.sum(pair_score_matrix)
            if num_grasp_pairs < min_num_grasps:
                print(f"Skipping {model_path} because it has {num_grasp_pairs} grasp pairs")
                continue
            
            # np.save(pair_score_matrix_path, pair_score_matrix)
            # np.save(pair_scores_path, pair_scores)
            # np.save(pair_grasps_path, tpp_grasp_dict)
            # np.save(point_cloud_path, point_cloud)
            triu_indices = np.triu_indices(1000, k=1)
            show_pair_edges(point_cloud, pair_scores, triu_indices, sample_info=sample)
            
        point_cloud_sample = TPPSample(model_path, point_cloud_path, pair_scores_path, pair_score_matrix_path,
                                        pair_grasps_path, grasp_file_path, sample)
        processed_mesh_count += 1
        point_cloud_samples.append(point_cloud_sample)

    return point_cloud_samples

if __name__ == "__main__":
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'
    #get obj files in in the directory
    num_points = 1000
    obj_files = os.listdir(model_root)
    # for file in obj_files:
    #     if file.endswith('.obj'):
    #         obj_path = os.path.join(model_root, file)
    #         mesh = o3d.io.read_triangle_mesh(obj_path)
    #         #scale the mesh
    #         # mesh.scale(scale, center=mesh.get_center())
    #         mesh.translate([-0.1, 0, -0.1], relative=False)

    #         o3d.utility.random.seed(0)
    #         point_cloud = mesh.sample_points_poisson_disk(num_points)
    #         o3d.visualization.draw_geometries([mesh, point_cloud])
    get_point_cloud_samples('../data', num_points=num_points)