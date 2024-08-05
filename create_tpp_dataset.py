import os
import numpy as np
from TPPSample import TPPSample
import tqdm
from acronym_utils import load_file_names, extract_sample_info
import h5py
import open3d as o3d


def save_split_samples(data_dir, num_mesh, train_ratio=0.8):
    if not os.path.exists('sample_dirs'):
        os.makedirs('sample_dirs')
    paths_dir = os.path.join('sample_dirs', 'success_tpp_samples.npy')
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
    pair_grasp_folder = os.path.join(data_dir, f'pair_grasps_{num_points}')
    point_cloud_folder = os.path.join(data_dir, f'point_cloud_{num_points}')
    touch_pair_score_folder = os.path.join(data_dir, f'touch_pair_scores_{num_points}')


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
        touch_pair_scores_path = f'{touch_pair_score_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        point_cloud_save_path = f'{point_cloud_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_grasps_save_path = f'{pair_grasp_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'

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
                simplified_mesh_count += 1
                success_grasp_mask = grasp_success > success_threshold
                success_grasp_poses = grasp_poses[success_grasp_mask]

                if not (os.path.exists(point_cloud_save_path) and os.path.exists(touch_pair_scores_path) and os.path.exists(pair_grasps_save_path)):
                    mesh_data = o3d.io.read_triangle_mesh(simplified_mesh_path)
                    point_cloud = mesh_data.sample_points_poisson_disk(num_points)
                    sample['scale'] = float(sample['scale'])
                    point_cloud = np.asarray(point_cloud.points) * sample['scale']


                    if not os.path.exists(touch_pair_scores_path) or not os.path.exists(pair_grasps_save_path):
                        pair_scores, tpp_grasp_dict = create_touch_point_pair_scores_and_grasps(point_cloud, success_grasp_poses, cylinder_radius=1, cylinder_height=1)
                        if np.sum(pair_scores) < min_num_grasps:
                            continue
                        np.save(touch_pair_scores_path, pair_scores)
                        np.save(pair_grasps_save_path, tpp_grasp_dict)

                    if not os.path.exists(point_cloud_save_path):
                        np.save(point_cloud_save_path, point_cloud)
                
                point_cloud_sample = TPPSample(simplified_mesh_path, point_cloud_save_path, touch_pair_scores_path, pair_grasps_save_path, grasps_file_name, sample)
                point_cloud_samples.append(point_cloud_sample)
    return point_cloud_samples


def create_touch_point_pair_scores_and_grasps(vertices, grasp_poses, cylinder_radius=0.02, cylinder_height=0.02, max_grasps_per_pair=20):
    gripper_right_tip_vector = np.array([-4.100000e-02, -7.27595772e-12, 1.12169998e-01, 1])
    gripper_left_tip_vector = np.array([4.10000000e-02, -7.27595772e-12, 1.12169998e-01, 1])
    right_tip_pos = np.matmul(grasp_poses, gripper_right_tip_vector)[:, :3]
    left_tip_pos = np.matmul(grasp_poses, gripper_left_tip_vector)[:, :3]
    tip_axis =  right_tip_pos - left_tip_pos
    tip_axis = tip_axis / np.linalg.norm(tip_axis)
    left_cylinder_bottom = left_tip_pos + tip_axis * cylinder_height
    right_cylinder_bottom = right_tip_pos - tip_axis * cylinder_height

    point_pair_score_matrix = np.zeros((vertices.shape[0], vertices.shape[0]), dtype=int)
    upper_tri_idx = np.triu_indices(vertices.shape[0], k=1)
    # tpp_grasps_matrix = np.zeros((vertices.shape[0], vertices.shape[0], max_grasps_per_pair, 4, 4))
    # grasp_counts = np.zeros((len(upper_tri_idx[0])))

    tpp_grasp_dict = {}
    for (i, j) in zip(upper_tri_idx[0], upper_tri_idx[1]):
        tpp_grasp_dict[frozenset((i, j))] = []

    # print(left_tip_pos.shape, left_cylinder_bottom.shape, right_tip_pos.shape, right_cylinder_bottom.shape, grasp_poses.shape)
    for (left_tip, left_bottom, right_tip, right_bottom, pose) in zip(left_tip_pos, left_cylinder_bottom, right_tip_pos, right_cylinder_bottom,  grasp_poses):
        # print(f"Left tip: {left_tip}, Left bottom: {left_bottom}, Right tip: {right_tip}, Right bottom: {right_bottom}")
        inside_right_cylinder = is_point_inside_cylinder(vertices, right_tip, right_bottom, cylinder_radius)
        inside_left_cylinder = is_point_inside_cylinder(vertices, left_tip, left_bottom, cylinder_radius)

        right_tip_touch_points_idxs = inside_right_cylinder.nonzero()[0]
        left_tip_touch_points_idxs = inside_left_cylinder.nonzero()[0]


        for i in left_tip_touch_points_idxs:
            for j in right_tip_touch_points_idxs:
                if i == j:
                    continue
                if point_pair_score_matrix[i, j] >= max_grasps_per_pair:
                    continue
 
                # tpp_grasps_matrix[i, j, point_pair_score_matrix[i, j]] = pose
                # tpp_grasps_matrix[j, i, point_pair_score_matrix[i, j]] = pose
                point_pair_score_matrix[i, j] += 1
                point_pair_score_matrix[j, i] += 1
                tpp_grasp_dict[frozenset((i, j))].append(pose)

    # print(point_pair_score_matrix)
    pair_scores = point_pair_score_matrix[upper_tri_idx]
    # tpp_grasps = tpp_grasps_matrix[upper_tri_idx]
    return pair_scores, tpp_grasp_dict

        

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
    gripper_right_tip_vector = np.array([-5.100000e-02, 0, 1.12169998e-01, 1])
    gripper_left_tip_vector = np.array([5.10000000e-02, 0, 1.12169998e-01, 1])
    cube_size = 0.08
    mid_point = (gripper_right_tip_vector[:3] + gripper_left_tip_vector[:3]) / 2
    points = np.random.uniform(-cube_size/2, cube_size/2, size=(N, 3)) + mid_point
    dummy_grasp_poses = np.eye(4).reshape(1, 4, 4).repeat(num_grasps, axis=0)
    for i in range(1, num_grasps):
        dummy_grasp_poses[i, :3, :3] = o3d.geometry.get_rotation_matrix_from_xyz([0, 0, i * np.pi/num_grasps])
    return points, dummy_grasp_poses

if __name__ == "__main__":
    # train_samples, val_samples = save_split_samples('../data', 100)
    # print(f"Number of train samples: {len(train_samples)}")
    # print(f"Number of validation samples: {len(val_samples)}")
    # print("Done!")

    # samp = train_samples[0]
    # print(samp)
    from gewa_dataset import GewaDataset
    import create_gewa_dataset

    # train_paths, val_paths = create_gewa_dataset.save_split_samples('../data', 100)
    # dataset = GewaDataset(train_paths, normalize_points=True)

    # sample = dataset[1]
    # points = sample.x.detach().numpy()
    # grasps = sample.y.detach().numpy()
    # num_grasps = sample.num_grasps.detach().numpy()
    # valid_grasps = grasps[num_grasps > 0]
    # p_idxs = np.random.randint(0, valid_grasps.shape[0], 2)
    # dummy_grasp_poses = valid_grasps[:10, 0]

    N = 1000

    num_grasps = 2
    points, dummy_grasp_poses = create_point_cloud_and_grasps(N, num_grasps)

    pair_scores, tpp_grasp_dict = create_touch_point_pair_scores_and_grasps(points, dummy_grasp_poses, cylinder_radius=0.01, cylinder_height=0.04)
    
    from acronym_visualize_dataset import visualize_grasps
    pair_idx = np.where(pair_scores > 0)[0]
    print(pair_idx)
    triu_indices = np.triu_indices(N, k=1)
    grasps = []
    contact_idxs = []
    for i, j in zip(triu_indices[0][pair_idx], triu_indices[1][pair_idx]):
        # print(len(tpp_grasp_dict[frozenset((i, j))]))
        # for grasp in tpp_grasp_dict[frozenset((i, j))]:
        #     grasps.append(grasp.reshape(4, 4))
        #     contact_idxs.append([i, j])
        # break
        grasps.append(tpp_grasp_dict[frozenset((i, j))][0].reshape(4, 4))
        contact_idxs.append([i, j])

    grasps = np.array(grasps)
    # print(grasps.shape)
    visualize_grasps(points, grasps, None, contact_idxs)
    visualize_grasps(points, dummy_grasp_poses, None, None)

    