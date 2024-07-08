from acronym_visualize_dataset import visualize_point_cloud, visualize_approach_points, visualize_grasps
from gewa_dataset import GewaDataset
from create_gewa_dataset import save_split_samples
import argparse
import numpy as np

def show_grasps(dataset, idx):
    point_idxs = np.random.randint(len(dataset[idx].pos), size=5)
    grasps = dataset[idx].y[point_idxs].numpy()
    contact_points_idx = dataset[idx].contact_points[point_idxs].numpy().astype(int)
    visualize_grasps(dataset[idx].pos.numpy(), grasps, point_idxs, contact_points_idx)


if __name__ == "__main__":
    train_data, valid_data = save_split_samples('../data', 100)
    dataset = GewaDataset(train_data)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index', type=int, default=0)

    args = parser.parse_args()
    idx = args.index
    print(dataset[idx])

    # print(dataset[0][1].shape)
    # print(dataset[0][2])
    # visualize_point_cloud(dataset[idx].pos)
    # visualize_approach_points(dataset[idx].pos.numpy(), dataset[idx].approach_scores.numpy())


    show_grasps(dataset, idx)