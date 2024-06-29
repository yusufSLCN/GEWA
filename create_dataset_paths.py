from acronym_utils import get_simplified_samples, get_simplified_meshes_w_closest_grasp
import numpy as np
import argparse
import os


def save_split_meshes(data_dir, num_mesh, train_ratio=0.8):
    if not os.path.exists('sample_dirs'):
        os.makedirs('sample_dirs')
    paths_dir = os.path.join('sample_dirs', 'success_simplified_meshes.npy')
    if os.path.exists(paths_dir):
        simplified_meshes = np.load(paths_dir, allow_pickle=True)

    else:
        simplified_meshes = get_simplified_meshes_w_closest_grasp(data_dir, num_mesh=-1)
        #sort by name 
        simplified_meshes = sorted(simplified_meshes, key=lambda x: x.split('/')[-1])
        np.random.seed(0)
        np.random.shuffle(simplified_meshes)
        np.save(paths_dir, simplified_meshes)

    #split samples into train and test sets
    if len(simplified_meshes) <= num_mesh or num_mesh < 0:
        num_mesh = len(simplified_meshes)
        print(f"Number of meshes in the simlified subset: {num_mesh}")

    split_idx = int(len(simplified_meshes) * train_ratio)
    all_train_meshes = simplified_meshes[:split_idx]
    all_valid_meshes = simplified_meshes[split_idx:]

    subset_idx = int(num_mesh * train_ratio)
    train_meshes = all_train_meshes[:subset_idx]
    valid_meshes = all_valid_meshes[:num_mesh - subset_idx]

    # #save the train and test meshes
    # np.save('sample_dirs/train_success_simplified_acronym_meshes.npy', train_meshes)
    # np.save('sample_dirs/valid_success_simplified_acronym_meshes.npy', valid_meshes)
    print(f"Train mesh {len(train_meshes)}")
    print(f"Test mesh {len(valid_meshes)}")
    return train_meshes, valid_meshes


def save_split_samples(data_dir, num_mesh, train_ratio=0.8):
    simplified_samples, mesh_sample_id_dict = get_simplified_samples(data_dir, num_mesh=num_mesh, override=True)
    #split samples into train and test sets
    mesh_names = list(mesh_sample_id_dict.keys())
    print(f"Number of meshes in the simlified subset: {len(mesh_names)}")
    subset_idx = int(len(mesh_names) * train_ratio)
    train_mesh_names = mesh_names[:subset_idx]
    valid_mesh_names = mesh_names[subset_idx:]
    train_samples = []
    valid_samples = []
    for mesh_name in train_mesh_names:
        for sampleId in mesh_sample_id_dict[mesh_name]:
            train_samples.append(simplified_samples[sampleId])

    for mesh_name in valid_mesh_names:
        for sampleId in mesh_sample_id_dict[mesh_name]:
            valid_samples.append(simplified_samples[sampleId])

    #save the train and test samples
    np.save('sample_dirs/train_success_simplified_acronym_samples.npy', train_samples)
    np.save('sample_dirs/valid_success_simplified_acronym_samples.npy', valid_samples)
    print(f"Train mesh {len(train_mesh_names)}")
    print(f"Test mesh {len(valid_mesh_names)}")
    print(f"Train pairs {len(train_samples)}")
    print(f"Test pairs {len(valid_samples)}")

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d','--data_dir', type=str, default='../data')
    parser.add_argument('-nm','--num_mesh', type=int, default=-1)
    parser.add_argument('-tr','--train_ratio', type=float, default=0.8)
    args = parser.parse_args()

    data_dir = args.data_dir
    num_mesh = args.num_mesh
    train_ratio = args.train_ratio
    # save_split_samples(data_dir, num_mesh, train_ratio)
    save_split_meshes(data_dir, num_mesh, train_ratio)