import numpy as np
from tpp_dataset import TPPDataset
from acronym_visualize_dataset import visualize_grasps
from create_tpp_dataset import save_split_samples
from tpp_dataset import TPPDataset
from acronym_visualize_dataset import visualize_grasps
import argparse
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index', type=int, default=0)
    parser.add_argument('-n','--num_grasps', type=int, default=5)
    args = parser.parse_args()
    train_samples, val_samples = save_split_samples('../data', 20)
    print(f"Number of train samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(val_samples)}")
    print("Done!")

    dataset = TPPDataset(train_samples)
    t = time.time()
    sample = dataset[args.index]
    print(f"Time taken: {time.time() - t}")
    pos =sample.pos.numpy()
    grasps_dict = sample.y
    pair_scores = sample.pair_scores.numpy()

    pair_idxs = np.where(pair_scores > 0)[0]
    random_idxs = np.random.randint(0, pair_idxs.shape[0], args.num_grasps)
    selected_pair_idxs = pair_idxs[random_idxs]
    print(selected_pair_idxs)
    selected_grasps = []
    contact_idxs = []

    for pair_idx in selected_pair_idxs:
        i, j = dataset.triu_indices[0][pair_idx], dataset.triu_indices[1][pair_idx]
        key = frozenset((i, j))
        selected_grasps.append(grasps_dict[key][0].reshape(4, 4))
        contact_idxs.append([i, j])

    visualize_grasps(pos, selected_grasps, None, contact_idxs)