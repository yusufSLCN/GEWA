from acronym_utils import get_simplified_samples
import numpy as np
import argparse



def save_split_samples(data_dir, num_mesh, train_ratio=0.8):
    simplified_samples, mesh_sampleId_dict = get_simplified_samples(data_dir, num_mesh=num_mesh)
    #split samples into train and test sets
    mesh_names = list(mesh_sampleId_dict.keys())
    print(f"Number of meshes: {len(mesh_names)}")
    train_mesh_names = mesh_names[:int(len(mesh_names) * train_ratio)]
    valid_mesh_names = mesh_names[int(len(mesh_names) * train_ratio):]
    train_samples = []
    valid_samples = []
    for mesh_name in train_mesh_names:
        for sampleId in mesh_sampleId_dict[mesh_name]:
            train_samples.append(simplified_samples[sampleId])

    for mesh_name in valid_mesh_names:
        for sampleId in mesh_sampleId_dict[mesh_name]:
            valid_samples.append(simplified_samples[sampleId])

    #save the train and test samples
    np.save('train_success_simplified_acronym_samples.npy', train_samples)
    np.save('valid_success_simplified_acronym_samples.npy', valid_samples)
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
    save_split_samples(data_dir, num_mesh, train_ratio)