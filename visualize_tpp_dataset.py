import numpy as np
from tpp_dataset import TPPDataset
from acronym_visualize_dataset import visualize_grasps
from create_tpp_dataset import save_split_samples
from tpp_dataset import TPPDataset
from acronym_visualize_dataset import visualize_grasps
import argparse
import time
from create_tpp_dataset import create_touch_point_pair_scores_and_grasps

def show_tpp_grasps(args, dataset, pos, grasps_dict, pair_scores):
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

def show_all_tpps_of_grasp(points, grasps_dict, pair_scores, dataset, args):
    pair_idxs = np.where(pair_scores > 0)[0]
    random_idxs = np.random.randint(0, pair_idxs.shape[0], args.num_grasps)
    selected_pair_idxs = pair_idxs[random_idxs]
    selected_grasps = []

    for pair_idx in selected_pair_idxs:
        i, j = dataset.triu_indices[0][pair_idx], dataset.triu_indices[1][pair_idx]
        key = frozenset((i, j))
        selected_grasps.append(grasps_dict[key][0].reshape(4, 4))
    
    point_pair_score_matrix, pair_scores, tpp_grasp_dict, cylinder_edges = create_touch_point_pair_scores_and_grasps(points, selected_grasps, cylinder_radius=0.01, cylinder_height=0.041)
    
    pair_idxs = np.where(point_pair_score_matrix > 0)
    print(pair_idxs)
    print("_"*100)
    grasps = []
    contact_idxs = []
    for i, j in zip(pair_idxs[0], pair_idxs[1]):
        grasps.append(tpp_grasp_dict[frozenset((i, j))][0].reshape(4, 4))
        contact_idxs.append([i, j])

    grasps = np.array(grasps)
    edges = []
    for cylinder in zip(*cylinder_edges):
        edges.append(cylinder)

    print("_"*100)
    print("contact_idxs", len(contact_idxs))
    visualize_grasps(points, grasps, None, contact_idxs, cylinder_edges=edges)


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

    # show_tpp_grasps(args, dataset, pos, grasps_dict, pair_scores)
    show_all_tpps_of_grasp(pos, grasps_dict, pair_scores, dataset, args)