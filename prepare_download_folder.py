
import os
from acronym_utils import load_file_names, extract_sample_info
import tqdm


def get_file_names(data_dir, success_threshold=0.5, num_mesh=-1, num_points=1000, min_num_grasps=100, radius=0.005):
    simplified_mesh_directory = os.path.join(data_dir, 'simplified_obj')
    grasp_directory =  os.path.join(data_dir, 'acronym/grasps')
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'
    pair_grasp_folder = os.path.join(data_dir, f'tpp_grasps_{num_points}_{radius}')
    point_cloud_folder = os.path.join(data_dir, f'tpp_point_cloud_{num_points}_{radius}')
    touch_pair_score_folder = os.path.join(data_dir, f'tpp_scores_{num_points}_{radius}')
    touch_pair_score_matrix_folder = os.path.join(data_dir, f'tpp_score_matrix_{num_points}_{radius}')

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
    import random
    idxs = random.sample(range(1, 1000), 5)

    selected_samples = sample_info[idxs]
    print("Extracting simplified meshes with closest grasps")
    for i, sample in tqdm.tqdm(enumerate(selected_samples), total=len(selected_samples)):
        simplified_mesh_path = f'{simplified_mesh_directory}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.obj'
        pair_scores_path = f'{touch_pair_score_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_score_matrix_path = f'{touch_pair_score_matrix_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        point_cloud_path = f'{point_cloud_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'
        pair_grasps_path = f'{pair_grasp_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'