import os
import h5py
import numpy as np
import pywavefront
from sklearn.neighbors import kneighbors_graph
import time

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

def get_simplified_samples(simplified_mesh_directory, success_threshold=0.5):
    grasp_directory = '../data/acronym/grasps'
    model_root = '../data/ShapeNetSem-backup/models-OBJ/models'

    grasp_file_names = load_file_names(grasp_directory)
    sample_paths = extract_sample_info(grasp_file_names, model_root=model_root)
    simplified_samples = []

    pos_sample_count = 0
    neg_sample_count = 0
    for i, sample in enumerate(sample_paths):
        if (i + 1) % 500 == 0:
            print(f"Processed {i+1}/{len(sample_paths)}")

        simplify_save_path = f'{simplified_mesh_directory}/{sample["class"]}_{sample["model_name"]}_{sample["scale"]}.obj'
        # Check if the simplified mesh exists
        if os.path.exists(simplify_save_path):
            sample["simplified_model_path"] = simplify_save_path
            grasps_file_name = sample['grasps']
            data = h5py.File(grasps_file_name, "r")
            grasp_poses = np.array(data["grasps/transforms"])
            grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
            # print(grasp_poses.shape, grasp_success.shape)

            for (pose, success) in zip(grasp_poses, grasp_success):
                if success > success_threshold:
                    pos_sample_count += 1
                    grasp_sample = sample
                    grasp_sample['grasp_pose'] = pose
                    grasp_sample['success'] = success
                    simplified_samples.append(grasp_sample)
                else:
                    neg_sample_count += 1
    print(f"Positive samples: {pos_sample_count}")
    print(f"Negative samples: {neg_sample_count}")
        


    return simplified_samples


def convert2graph(sample, N=None):
    if N is None or N >= len(sample):
        simplified_mesh = sample
    else:
        idx = np.random.choice(len(sample), N, replace=False)
        simplified_mesh = sample[idx]

    A = kneighbors_graph(simplified_mesh, 6, mode='connectivity', include_self=True)
    return simplified_mesh, A

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
