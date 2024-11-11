from acronym_visualize_dataset import visualize_point_cloud, visualize_approach_points, visualize_grasps
from gewa_dataset import GewaDataset
from create_gewa_dataset import save_contactnet_split_samples
import argparse
import numpy as np

def show_grasps(dataset, idx, show_contacts=False):
    approach_scores = dataset[idx].approach_scores.numpy()
    good_approach_score_idxs = np.arange(approach_scores.shape[0])[approach_scores > 0.5]
    point_idxs = np.random.choice(good_approach_score_idxs, 3)
    print(point_idxs)
    # point_idxs = np.random.randint(len(dataset[idx].pos), size=5)
    # point_idxs = [0] * 10
    grasps = dataset[idx].y[point_idxs, 0].numpy().reshape(-1, 4, 4)
    print(grasps.shape)
    if show_contacts:
        contact_points_idx = dataset[idx].contact_points[point_idxs].numpy().astype(int)
    else:
        contact_points_idx = None

    visualize_grasps(dataset[idx].pos.numpy(), grasps, point_idxs, contact_points_idx, show_tip=False)


def show_object_graph(sample, ratio, r):
    from torch_geometric.nn import fps, radius
    import torch
    import open3d as o3d

    pos = sample.pos
    batch = torch.zeros(pos.shape[0], dtype=torch.long)

    idx = fps(pos, batch, ratio=ratio)
    row, col = radius(pos, pos[idx], r, batch, batch[idx],
                          max_num_neighbors=16)
    edge_index = torch.stack([col, row], dim=0).numpy().T
    points = pos.numpy()
    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(edge_index))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos.numpy())
    o3d.visualization.draw_geometries([line_set, pcd])

def show_knn_grap(sample, k=16):
    from torch_geometric.nn import knn
    import torch
    pos = sample.pos
    batch = torch.zeros(pos.shape[0], dtype=torch.long)
    row, col = knn(pos, pos, k, batch, batch)
    edge_index = torch.stack([col, row], dim=0).numpy().T
    points = pos.numpy()
    import open3d as o3d
    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(edge_index))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pos.numpy())
    o3d.visualization.draw_geometries([line_set, pcd])






if __name__ == "__main__":
    # train_data, valid_data = save_split_samples('../data', -1)
    #train samples: banana 100, bear bottle 160, bread slice 250
    #
    train_samples, val_samples = save_contactnet_split_samples('../data', num_mesh=1200)
    dataset = GewaDataset(train_samples)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index', type=int, default=0)

    args = parser.parse_args()
    idx = args.index
    # visualize_point_cloud(dataset[idx].pos)
    for i in range(len(dataset)):
        if dataset[i].sample_info['class'] == 'Cup':
            idx = i
            break
    print(dataset[idx].sample_info)
    visualize_approach_points(dataset[idx].pos.numpy(), dataset[idx].approach_scores.numpy())
    # show_object_graph(dataset[idx], ratio=0.2, r=0.0001)
    show_grasps(dataset, idx, show_contacts=False)
    # show_knn_grap(dataset[idx])