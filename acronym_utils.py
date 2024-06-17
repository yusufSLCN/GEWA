import os
import h5py
import numpy as np
import pywavefront
from sklearn.neighbors import kneighbors_graph
import time
import tqdm
import trimesh

def load_file_names(directory):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(os.path.join(root, file))
    return file_names



def extract_sample_info(file_names, model_root):
    samples_paths = []
    for file_name in file_names:
        base_name = os.path.basename(file_name)
        base_name = os.path.splitext(base_name)[0]
        name_part = base_name.split('_')
        model_file_path = os.path.join(model_root, name_part[1] + '.obj')
        sample = {'class' : name_part[0], 'grasps': file_name, 'model_path' : model_file_path , 'model_name': name_part[1], 'scale' : name_part[2] } 
        samples_paths.append(sample)
    return samples_paths

        
def load_sample(sample):
    grasp_file_name = sample['grasps']
    data = h5py.File(grasp_file_name, "r")
    T = np.array(data["grasps/transforms"])
    success = np.array(data["grasps/qualities/flex/object_in_gripper"])

    # Load the model
    model_file_name = sample['simplified_model_path']
    mesh_scale = sample['scale']

    mesh_data = pywavefront.Wavefront(model_file_name)
    vertices = np.array(mesh_data.vertices)
    
    return T, success, vertices

# this function returns the simplified samples and the dictionary that maps the mesh file path to the sample ids
def get_simplified_samples(data_dir, success_threshold=0.5, num_mesh=-1, override=False):
    simplified_mesh_directory = os.path.join(data_dir, 'simplified_obj')
    grasp_directory =  os.path.join(data_dir, 'acronym/grasps')
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'
    point_grasp_dict_folder = os.path.join(data_dir, 'point_grasp_dict')
    if not os.path.exists(point_grasp_dict_folder):
        print(f"Creating directory {point_grasp_dict_folder}")
        os.makedirs(point_grasp_dict_folder)

    grasp_file_names = load_file_names(grasp_directory)
    sample_paths = extract_sample_info(grasp_file_names, model_root=model_root)
    simplified_samples = []
    # Dictionary to store the closest grasps to each point in the mesh
    mesh_sample_id_dict = {}

    pos_sample_count = 0
    neg_sample_count = 0
    simplified_mesh_count = 0
    for i, sample in enumerate(sample_paths):
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(sample_paths)}")

        simplify_save_path = f'{simplified_mesh_directory}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.obj'
        point_grasp_save_path = f'{point_grasp_dict_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'

        # Check if the simplified mesh exists because not all samples have been simplified
        if os.path.exists(simplify_save_path):
            if num_mesh > 0 and simplified_mesh_count >= num_mesh:
                print(f"Positive samples: {pos_sample_count}")
                print(f"Negative samples: {neg_sample_count}")
                return simplified_samples, mesh_sample_id_dict
            
            sample["simplified_model_path"] = simplify_save_path
            grasps_file_name = sample['grasps']
            data = h5py.File(grasps_file_name, "r")
            grasp_poses = np.array(data["grasps/transforms"])
            grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
            sample["point_grasp_save_path"] = point_grasp_save_path


            #Filter the grasps based on the success threshold and add them to the simplified samples
            any_success = False
            for (pose, success) in zip(grasp_poses, grasp_success):
                if success > success_threshold:
                    grasp_sample = sample
                    grasp_sample['grasp_pose'] = pose
                    grasp_sample['success'] = success
                    simplified_samples.append(grasp_sample)
                    if simplify_save_path in mesh_sample_id_dict:
                        mesh_sample_id_dict[simplify_save_path].append(pos_sample_count)
                    else:
                        mesh_sample_id_dict[simplify_save_path] = [pos_sample_count]
                    pos_sample_count += 1
                    any_success = True
                else:
                    neg_sample_count += 1
            
            if any_success:
                simplified_mesh_count += 1
                # Check if the point grasp dict exists and create it if it doesn't
                if not os.path.exists(point_grasp_save_path) or override:
                    mesh_data = pywavefront.Wavefront(simplify_save_path)
                    vertices = np.array(mesh_data.vertices) * float(sample["scale"])
                    point_grasp_dict = create_point_grasp_dict(vertices, grasp_poses, grasp_success)
                    np.save(point_grasp_save_path, point_grasp_dict)
            else:
                print(f"No successful grasps for {simplify_save_path} in {grasp_poses.shape[0]} grasps")

    print(f"Number of meshes in the dataset: {len(sample_paths)}")
    print(f"Positive samples: {pos_sample_count}")
    print(f"Negative samples: {neg_sample_count}")


    return simplified_samples, mesh_sample_id_dict

def get_simplified_meshes_w_closest_grasp(data_dir, success_threshold=0.5, num_mesh=-1, n=5):
    simplified_mesh_directory = os.path.join(data_dir, 'simplified_obj')
    grasp_directory =  os.path.join(data_dir, 'acronym/grasps')
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'
    point_grasp_dict_folder = os.path.join(data_dir, 'point_grasp_dict')
    if not os.path.exists(point_grasp_dict_folder):
        print(f"Creating directory {point_grasp_dict_folder}")
        os.makedirs(point_grasp_dict_folder)

    grasp_file_names = load_file_names(grasp_directory)
    sample_paths = extract_sample_info(grasp_file_names, model_root=model_root)
    simplified_samples = []
    # Dictionary to store the closese grasps to each point in the mesh
    simplified_mesh_count = 0
    print("Extracting simplified meshes with closest grasps")
    for i, sample in tqdm.tqdm(enumerate(sample_paths), total=len(sample_paths)):
        simplify_save_path = f'{simplified_mesh_directory}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.obj'
        point_grasp_save_path = f'{point_grasp_dict_folder}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.npy'

        # Check if the simplified mesh exists because not all samples have been simplified
        if os.path.exists(simplify_save_path):
            if num_mesh > 0 and simplified_mesh_count >= num_mesh:
                return simplified_samples
            
            grasps_file_name = sample['grasps']
            data = h5py.File(grasps_file_name, "r")
            grasp_poses = np.array(data["grasps/transforms"])
            grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
            num_success = np.sum(grasp_success > success_threshold)
            if num_success > n:
                simplified_mesh_count += 1
                sample["simplified_model_path"] = simplify_save_path
                sample["point_grasp_save_path"] = point_grasp_save_path
                simplified_samples.append(sample)
                # Check if the point grasp dict exists and create it if it doesn't
                if not os.path.exists(point_grasp_save_path):
                    mesh_data = pywavefront.Wavefront(simplify_save_path)
                    vertices = np.array(mesh_data.vertices, dtype=np.float32) * float(sample["scale"])
                    point_grasp_dict = create_point_grasp_dict(vertices, grasp_poses, grasp_success, n=n)
                    np.save(point_grasp_save_path, point_grasp_dict)
    return simplified_samples
            
def find_n_closest_grasps(querry_point, grasp_poses, grasp_success, n=5):
    success_idxs = np.where(grasp_success > 0.5)
    success_grasp_poses = grasp_poses[success_idxs]
    grasp_success = grasp_success[success_idxs]
    if success_grasp_poses.shape[1:] != (4, 4):
        success_grasp_poses = success_grasp_poses.reshape(-1, 4, 4)
    # success_prasp_locations = np.array([0, 0, 0, 1])
    #tip of the gripper
    success_prasp_locations = np.array([0, 0, 1.12169998e-01, 1])
    
    success_grasp_locations = np.matmul(success_grasp_poses, success_prasp_locations)[:, :3]
    distances = np.linalg.norm(querry_point - success_grasp_locations, axis=1)
    sorted_indices = np.argsort(distances)
    closest_grasps = []
    try:
        for i in range(n):
            closest_grasps.append((success_grasp_poses[sorted_indices[i]], grasp_success[sorted_indices[i]]))
    except:
        print(f"Warning: Not enough succesful grasps: {success_grasp_locations.shape}, {np.sum(grasp_success == 1)}, {sorted_indices.shape}")
    return closest_grasps

def create_point_grasp_dict(vertices, grasp_poses, grasp_success, n=5):
    point_grasp_dict = {}
    for i in range(vertices.shape[0]):
        point = vertices[i].astype(np.float32)
        point_key = tuple(np.round(point, 3))
        closest_grasps = find_n_closest_grasps(point, grasp_poses, grasp_success, n=n)
        point_grasp_dict[point_key] = closest_grasps
    return point_grasp_dict

def convert2graph(sample, N=None):
    if N is None or N >= len(sample):
        simplified_mesh = sample
    else:
        idx = np.random.choice(len(sample), N, replace=False)
        simplified_mesh = sample[idx]

    A = kneighbors_graph(simplified_mesh, 6, mode='connectivity', include_self=True)
    return simplified_mesh, A

def create_gripper_marker(color=[0, 0, 255], tube_radius=0.001, sections=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.

    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]]
    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
    )

    tmp = trimesh.util.concatenate([cb1, cb2, cfr, cfl])
    tmp.visual.face_colors = color

    return tmp


def analyze_dataset_stats(dataset):
    train_class_stats = {}
    #analyze the dataset
    for i in range(len(dataset)):
        sample = dataset[i]
        sample_info = sample[2]
        sample_class = sample_info["class"]
        if sample_class in train_class_stats:
            train_class_stats[sample_class] += 1
        else:
            train_class_stats[sample_class] = 1

    return train_class_stats

if __name__ == "__main__":
    grasp_directory = '/Users/yusufsalcan/Documents/CS_Semester_2/Grasp_Everything_with_Anything/grasps'
    model_root = '/Users/yusufsalcan/Documents/CS_Semester_2/Grasp_Everything_with_Anything/data/ShapeNetSem-backup/models-OBJ/models'

    grasp_file_names = load_file_names(grasp_directory)
    sample_paths = extract_sample_info(grasp_file_names, model_root=model_root)
    start_time = time.time()

    # Loop code here
    T, success, mesh = load_sample(sample_paths[0])
    # print(T)
    mesh, A = convert2graph(mesh, N=1000)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time)


