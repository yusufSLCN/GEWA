from torch_geometric.datasets import ShapeNet
import torch_geometric.transforms as T
from utils import plot_point_cloud, plot_network
import numpy as np
from gewa_dataset import GewaDataset

if __name__ == "__main__":
    # dataset = ShapeNet(root='data/ShapeNet', categories=['Mug'], split='val', pre_transform=T.KNNGraph(k=6))
    from gewa_dataset import GewaDataset
    from create_gewa_dataset import save_split_samples

    train_paths, val_paths = save_split_samples('../data', -1)
    dataset = GewaDataset(train_paths, normalize_points=True)

    dataset_average_distance = []

    for i in range(len(dataset)):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(dataset)}")

        sample = dataset[i]
        pos = sample.pos.numpy()
        approach_scores = sample.approach_scores.numpy()
        sample_grasps = sample.y.numpy()
        num_grasps = sample.num_grasps.numpy()
        point_average = []
        for j in range(len(pos)):
            if num_grasps[j] > 0:
                point = pos[j]
                grasps = sample_grasps[j, :num_grasps[j]]
                gripper_tip_vector = np.array([0, 0, 1.12169998e-01, 1])
                grasp_tip_pos = np.matmul(grasps, gripper_tip_vector)[:, :3]
                #calculate the distance between the point and grasp tip
                distances = np.linalg.norm(grasp_tip_pos - point, axis=1)

                mean_point_distance = np.mean(distances)
                if mean_point_distance > 0.01:
                    print(distances)
                    print(approach_scores[j])
                    print(f"{sample.sample_info}")
                    print(f"Object {i}, point {j}")
                    print(f"Mean point distance: {mean_point_distance}")
                    #stop code execution
                    # exit()
                point_average.append(mean_point_distance)

        dataset_average_distance.append(np.array(point_average).mean())

    print(f"Average distance: {np.array(dataset_average_distance).mean()}")

