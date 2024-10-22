from create_tpp_dataset import get_contactnet_split
import os
import numpy as np
import json
from acronym_utils import extract_sample_info_from_basename
import h5py

def create_contactnet_splits(num_mesh=400, min_num_success_grasps=100):

    train_save_file = f"contactnet_{num_mesh}_train_split.npy"
    valid_save_file = f"contactnet_{num_mesh}_valid_split.npy"

    if os.path.exists(train_save_file) and os.path.exists(valid_save_file):
        return train_save_file, valid_save_file

    train_meshes, valid_meshes = get_contactnet_split()
    simplified_mesh_directory = '../data/simplified_obj'
    simplified_meshes= os.listdir(simplified_mesh_directory)
    simplified_meshes = sorted(simplified_meshes)
    for file in simplified_meshes:
        if not file.endswith(".obj"):
            simplified_meshes.remove(file)
        
    #remove file extension
    simplified_meshes = [os.path.splitext(file)[0] for file in simplified_meshes]
    print(f"{len(simplified_meshes)} simplified meshes")
    print(simplified_meshes[0])

    # dataset_name = "tpp_effdict_nomean_wnormals"
    # radius=0.005
    # num_mesh = 400
    # paths_dir = os.path.join('sample_dirs', f'{dataset_name}_r-{radius}_{num_mesh}_samples.npy')
    # print(paths_dir)
    # if os.path.exists(paths_dir):
    #     samples = np.load(paths_dir, allow_pickle=True)
    # else:
    #     print("No samples found")

    # simplified_meshes = [os.path.splitext(os.path.basename(sample.simplified_mesh_path))[0] for sample in samples]
    # print(f"{len(simplified_meshes)} simplified meshes")
    # print(simplified_meshes[0])
    data_dir = '../data'
    grasp_directory =  os.path.join(data_dir, 'acronym/grasps')

    path = 'splits'
    train_meshes = []
    valid_meshes = []
    split_paths = os.listdir(path)
    split_paths = sorted(split_paths)
    max_train_class_samples = 16
    max_test_class_samples = 4
    #json files in the path
    for model_split_file in split_paths:
        if len(train_meshes) + len(valid_meshes) >= num_mesh:
            break

        if model_split_file.endswith(".json"):
            with open(os.path.join(path, model_split_file), 'r') as file:
                data = json.load(file)

                train_names = [os.path.splitext(file_name)[0] for file_name in data['train']]
                test_names = [os.path.splitext(file_name)[0] for file_name in data['test']]

                num_simplified_train = 0
                num_simplified_test = 0
                for name in train_names:
                    if name in simplified_meshes:
                        grasp_file_path = os.path.join(grasp_directory, name + '.h5')
                        num_success = get_num_success_grasps(grasp_file_path)
                        if num_success < min_num_success_grasps:
                            continue

                        if num_simplified_train >= max_train_class_samples:
                            break
                        num_simplified_train += 1
                        train_meshes.append(name)

                for name in test_names:
                    if name in simplified_meshes:
                        grasp_file_path = os.path.join(grasp_directory, name + '.h5')
                        num_success = get_num_success_grasps(grasp_file_path)
                        if num_success < min_num_success_grasps:
                            continue
                        if num_simplified_test >= max_test_class_samples:
                            break
                        num_simplified_test += 1
                        valid_meshes.append(name)
                # print(f"Model: {model_split_file}")
                # print(f"Train: {num_simplified_train} simplified meshes")
                # print(f"Test: {num_simplified_test} simplified meshes")

    print(f"Train: {len(train_meshes)} simplified meshes")
    print(f"Test: {len(valid_meshes)} simplified meshes")

    np.save(train_save_file, train_meshes)
    np.save(valid_save_file, valid_meshes)

    return train_save_file, valid_save_file


def get_num_success_grasps(grasps_file_name, success_threshold=0.5):
    data = h5py.File(grasps_file_name, "r")
    grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    num_success = np.sum(grasp_success > success_threshold)
    return num_success
    

if __name__ == "__main__":
    create_contactnet_splits(1000)
    # num_mesh = 1000
    # train_meshes = np.load(f"contactnet_{num_mesh}_train_split.npy")
    # print(train_meshes)
