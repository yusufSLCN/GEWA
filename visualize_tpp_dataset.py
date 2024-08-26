import numpy as np
import open3d as o3d
from matplotlib import colormaps
import matplotlib.pyplot as plt
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

def show_all_tpps_of_grasp(points, grasps_dict, pair_scores, triu_indices, args):
    pair_idxs = np.where(pair_scores > 0)[0]
    random_idxs = np.random.randint(0, pair_idxs.shape[0], args.num_grasps)
    selected_pair_idxs = pair_idxs[random_idxs]
    selected_grasps = []

    for pair_idx in selected_pair_idxs:
        i, j = triu_indices[0][pair_idx], triu_indices[1][pair_idx]
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

def show_pair_edges(points, pair_scores, triu_indices, sample_info=None, threshold=0.5):
    pair_idxs = np.where(pair_scores > threshold)[0]
    good_pair_scores = pair_scores[pair_idxs]
    edge_index = np.stack((triu_indices[0][pair_idxs], triu_indices[1][pair_idxs]), axis=1) 
    cmap_name = 'viridis'  # You can change this to any other colormap name
    cmap = colormaps[cmap_name]
    norm = plt.Normalize(good_pair_scores.min(), good_pair_scores.max())
    colors = cmap(norm(good_pair_scores))[:, :3]
    print(good_pair_scores.min(), good_pair_scores.max())

    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(edge_index))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.estimate_normals()
    # print(np.asarray(pcd.normals).shape)
    gripper = create_gripper()
    if sample_info is not None:
        obj_path = sample_info['model_path']
        scale = float(sample_info['scale'])
        mesh = o3d.io.read_triangle_mesh(obj_path)
        #scale the mesh
        mesh.scale(scale, center=mesh.get_center())
        mesh.translate([-0.1, 0, -0.1], relative=False)
        o3d.visualization.draw_geometries([mesh, line_set, pcd, gripper])
        
    else:
        o3d.visualization.draw_geometries([line_set, pcd, gripper])


def create_gripper(color=[0, 0, 255], tube_radius=0.001, resolution=6):
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.

    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        resolution (int, optional): Resolution of cylinders. Defaults to 6.

    Returns:
        open3d.geometry.TriangleMesh: A mesh that represents a simple parallel yaw gripper.
    """
    def create_cylinder(p1, p2, radius, resolution):
        p1 = np.array(p1)
        p2 = np.array(p2)
        vector = p2 - p1
        length = np.linalg.norm(vector)
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
        cylinder.compute_vertex_normals()
        
        # Rotate and translate cylinder
        rotation = o3d.geometry.get_rotation_matrix_from_xyz((0, np.arccos(vector[2] / length), 0))
        cylinder.rotate(rotation, center=(0, 0, 0))
        cylinder.translate((p1 + p2) / 2)
        
        return cylinder

    cfl = create_cylinder([4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
                          [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
                          tube_radius, resolution)
    
    cfr = create_cylinder([-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
                          [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
                          tube_radius, resolution)
    
    cb1 = create_cylinder([0, 0, 0], 
                          [0, 0, 6.59999996e-02],
                          tube_radius, resolution)
    
    cb2 = create_cylinder([-4.100000e-02, 0, 6.59999996e-02],
                          [4.100000e-02, 0, 6.59999996e-02],
                          tube_radius, resolution)

    # Combine all cylinders
    gripper_mesh = cfl + cfr + cb1 + cb2

    # Set color
    color_array = np.array(color) / 255.0  # Normalize color values to [0, 1]
    gripper_mesh.paint_uniform_color(color_array)

    return gripper_mesh

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--index', type=int, default=0)
    parser.add_argument('-n','--num_grasps', type=int, default=5)
    args = parser.parse_args()
    train_samples, val_samples = save_split_samples('../data', 100)
    print(f"Number of train samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(val_samples)}")
    print("Done!")

    dataset = TPPDataset(train_samples, return_pair_dict=True)
    t = time.time()
    sample = dataset[args.index]
    print(sample.sample_info)
    pos =sample.pos.numpy()
    grasps_dict = sample.y
    pair_scores = sample.pair_scores.numpy()

    # show_tpp_grasps(args, dataset, pos, grasps_dict, pair_scores)
    # show_all_tpps_of_grasp(pos, grasps_dict, pair_scores, dataset.triu_indices, args)
    show_pair_edges(pos, pair_scores, dataset.triu_indices, sample.sample_info)