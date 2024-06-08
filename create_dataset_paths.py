from acronym_utils import get_simplified_samples
import numpy as np

simplified_mesh_directory = '../data/simplified_obj'
simplified_samples = get_simplified_samples(simplified_mesh_directory)
#split samples into train and test sets

print(f"Number of meshes: {len(simplified_samples)/2000}" )
split_idx = int((len(simplified_samples)/2000) * 0.8) * 2000
train_samples = simplified_samples[:int(split_idx)]
valid_samples = simplified_samples[int(split_idx):]
#save the train and test samples
np.save('train_success_simplified_acronym_samples.npy', train_samples)
np.save('valid_success_simplified_acronym_samples.npy', valid_samples)
print(f"Train {len(train_samples)}")
print(f"Test {len(valid_samples)}")
print(len(simplified_samples))