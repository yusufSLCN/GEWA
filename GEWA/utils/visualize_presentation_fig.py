import numpy as np
import open3d as o3d
from matplotlib import colormaps
import matplotlib.pyplot as plt
from tpp_dataset import TPPDataset
from visualization.visualize_acronym_dataset import visualize_grasps
from create_tpp_dataset import save_contactnet_split_samples
from tpp_dataset import TPPDataset
import argparse
import time
from create_tpp_dataset import create_touch_point_pair_scores_and_grasps
import os

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

def show_grasp_and_edge_predictions(points, gt_grasp_dict, selected_edge_idx, grasp_pred, triu_indices, sample_info=None, num_grasp_to_show=5, mean =0,
                                    view_params=None, save_name=None, save=False, display_obj=False):
    gt_gripper_meshes = []
    pred_gripper_meshes = []
    selected_edge_idx = selected_edge_idx[:num_grasp_to_show]
    
    for grasp_idx, edge_idx in enumerate(selected_edge_idx):
        i = triu_indices[0][edge_idx]
        j = triu_indices[1][edge_idx]
        key = frozenset((i, j))
        if key in gt_grasp_dict:
            gt_gripper_mesh = create_gripper(color=[0, 255, 0])
            trans = gt_grasp_dict[key][0].reshape(4, 4)
            trans[0:3, 3] -= mean
            # gt_gripper_mesh.transform(gt_grasp_dict[key][0].reshape(4, 4))
            gt_gripper_meshes.append(gt_gripper_mesh)

        pred_gripper_mesh = create_gripper()
        pred_gripper_mesh.transform(grasp_pred[grasp_idx].reshape(4, 4))
        pred_gripper_meshes.append(pred_gripper_mesh)
    
    
    # pair_idxs = np.where(pair_scores > 0)[0]
    # random_idxs = np.random.randint(0, pair_idxs.shape[0], args.num_grasps)
    # selected_pair_idxs = pair_idxs[random_idxs]
    selected_edge_index = np.stack((triu_indices[0][selected_edge_idx], triu_indices[1][selected_edge_idx]), axis=1) 

    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(selected_edge_index))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    #show the mesh
    if display_obj:
        obj_path = sample_info['model_path'][0]
        scale = float(sample_info['scale'][0])
        mesh = o3d.io.read_triangle_mesh(obj_path)
        #scale the mesh
        mesh.scale(scale, center=mesh.get_center())
        mesh.translate([0, 0.2, 0], relative=False)
        mesh.compute_vertex_normals()
        meshes = [mesh, line_set, pcd, *pred_gripper_meshes]
        
    else:
        meshes = [line_set, pcd, *pred_gripper_meshes]

    if view_params is None:
        view_params = {
        'zoom': 1,
        'front': [1, -0.5, -1],    # Camera direction
        'up': [0, 1, 0],         # Up direction
        'lookat': pcd.get_center()
        }
    view_params['lookat'] = pcd.get_center()

    display_mesh(meshes, view_params, save_name=save_name, save_image=save)

def show_grasps_of_edges(points, grasps_dict, pair_scores, triu_indices, args, sample_info=None, mean=None, view_params=None, save=False):
    pair_idxs = np.where(pair_scores > 0)[0]
    selected_pair_idxs = pair_idxs[0:1]
    # random_idxs = np.random.randint(0, pair_idxs.shape[0], args.num_grasps)
    # selected_pair_idxs = pair_idxs[random_idxs]

    gripper_meshes = []
    for pair_idx in selected_pair_idxs:
        i, j = triu_indices[0][pair_idx], triu_indices[1][pair_idx]
        key = frozenset((i, j))
        gripper_mesh = create_gripper([0, 255, 0])

        transform = grasps_dict[key][0].reshape(4, 4)
        if mean is not None:
            transform[0:3, 3] -= mean
        # gripper_mesh.transform(transform)
        # gripper_meshes.append(gripper_mesh)

        pred_gripper_mesh = create_gripper()
        #rotate 30 degrees
        angle = np.pi/6
        # transform[:3, :3] = transform[:3, :3] @ np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])
        #rotate 30 degrees around z axis
        transform[:3, :3] = transform[:3, :3] @ np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        #shift 3cm in z direction
        transform[0:3, 3] += np.array([0, 0, 0.01])
        pred_transform = transform
        pred_gripper_mesh.transform(pred_transform)
        gripper_meshes.append(pred_gripper_mesh)


    selected_edge_index = np.stack((triu_indices[0][selected_pair_idxs], triu_indices[1][selected_pair_idxs]), axis=1) 

    line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(selected_edge_index))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if view_params is None:
        view_params = {
        'zoom': 1,
        'front': [1, -0.5, -1],    # Camera direction
        'up': [0, 1, 0],         # Up direction
        'lookat': pcd.get_center()
        }
    view_params['lookat'] = pcd.get_center()
    

    if sample_info is not None: 
        save_name = sample_info['class'] +  "_grasps.png"
    else:
        save_name = "grasps.png"

    mesh = [line_set, pcd, *gripper_meshes]
    display_mesh(mesh, view_params, save_name=save_name, save_image=save)


def read_obj_mesh(sample_info):
    obj_path = sample_info['model_path']
    if not isinstance(obj_path, str):
        obj_path = obj_path[0]
    scale = float(sample_info['scale'])
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    #scale the mesh
    mesh.scale(scale, center=mesh.get_center())
    return mesh

def show_obj_mesh(sample_info, view_params=None, save_name="", save=False):
    mesh = read_obj_mesh(sample_info)
    # o3d.visualization.draw_geometries([mesh])

    if view_params is None:
        view_params = {
        'zoom': 1,
        'front': [1, 0.5, -1],    # Camera direction
        'up': [0, 1, 0],         # Up direction
        'lookat': mesh.get_center()
        }
    
    view_params['lookat'] = mesh.get_center()
    
    display_mesh(mesh, view_params, save_name=save_name, save_image=save)


def display_mesh(mesh, view_params, save_name=None, save_image=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if isinstance(mesh, list):
        for m in mesh:
            vis.add_geometry(m)
    else:
        vis.add_geometry(mesh)
    view_control = vis.get_view_control()
    view_control.set_lookat(view_params['lookat'])
    view_control.set_front(view_params['front'],)
    view_control.set_up(view_params['up'])
    view_control.set_zoom(view_params['zoom'])

    if save_image:
        vis.poll_events()
        vis.update_renderer()
        # Capture and save the image
        image = vis.capture_screen_float_buffer(False)  # Set True for float buffer
        image_np = np.asarray(image)
        
        # Convert to uint8 format (0-255 range)
        image_np = (image_np * 255).astype(np.uint8)
        
        # Create Open3D Image
        o3d_image = o3d.geometry.Image(image_np)
        
        save_path = os.path.join("images", save_name)
        # Save the image
        success = o3d.io.write_image(save_path, o3d_image)
        print(f"Image saved successfully: {success}")
        vis.destroy_window()
    else:
        vis.run()

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

def show_pair_edges(points, pair_scores, triu_indices, threshold=0.5, show_gripper=False, view_params=None, save_name="", save=False):
    pair_idxs = np.where(pair_scores > threshold)[0]
    #select only one idx
    pair_idxs = pair_idxs[0:1]
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
    pcd.estimate_normals()
    # print(np.asarray(pcd.normals).shape)
    if show_gripper:
        gripper = create_gripper()
    else:
        gripper = o3d.geometry.TriangleMesh()

    if view_params is None:
        view_params = {
        'zoom': 1,
        'front': [1, -0.5, -1],    # Camera direction
        'up': [0, 1, 0],         # Up direction
        'lookat': pcd.get_center()
        }

    view_params['lookat'] = pcd.get_center()

    if save and save_name == "":
        save_name = "edge.png"


    mesh = [line_set, pcd, gripper]
    display_mesh(mesh, view_params, save_name=save_name, save_image=save)
    # if sample_info is not None:
    #     mesh = read_obj_mesh(sample_info)
    #     mesh.translate([-0.1, 0, -0.1], relative=False)
    #     o3d.visualization.draw_geometries([mesh, line_set, pcd, gripper], 
    #                                         zoom=view_params['zoom'],
    #                                         front=view_params['front'],
    #                                         lookat=mesh.get_center(),
    #                                         up=view_params['up'])
        
    # else:
    #     o3d.visualization.draw_geometries([line_set, pcd, gripper], 
    #                                         zoom=view_params['zoom'],
    #                                         front=view_params['front'],
    #                                         lookat=pcd.get_center(),
    #                                         up=view_params['up'])


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
    parser.add_argument('-s', '--save', action='store_true')
    args = parser.parse_args()
    train_samples, val_samples = save_contactnet_split_samples('../data', 1200, dataset_name="tpp_effdict_nomean_wnormals")
    print(f"Number of train samples: {len(train_samples)}")
    print(f"Number of validation samples: {len(val_samples)}")
    print("Done!")
    #train samples: banana 100, bear bottle 160, bread slice 250, 650 mug
    #valid samples: desk 160, curtain 155, fork 232, foodbag 230 ------ try 224, 237, 248
    #42 teddy bear, 75 bowl

    dataset = TPPDataset(val_samples, return_pair_dict=True, normalize=True)
    sample = dataset[args.index]
    print(sample.sample_info)
    pos =sample.pos.numpy()
    grasps_dict = sample.y
    pair_scores = sample.pair_scores.numpy()
    mean = sample.sample_info["mean"]

    view_params = {
    'zoom': 0.8,
    'front': [1, 0.5, 1],    # Camera direction
    'up': [0, 0, 1]         # Up direction
    }
    # show_tpp_grasps(args, dataset, pos, grasps_dict, pair_scores)
    # show_all_tpps_of_grasp(pos + mean, grasps_dict[0], pair_scores, dataset.triu_indices, args)
    save_image = args.save
    # show_obj_mesh(sample.sample_info, view_params, save=save_image)
    # show_grasps_of_edges(pos, grasps_dict[0], pair_scores, dataset.triu_indices, args, sample_info=sample.sample_info,
    #                       mean=mean, view_params=view_params, save=save_image)
    # show_pair_edges(pos, pair_scores, dataset.triu_indices,
    #                  view_params=view_params, save=save_image)

    for i in range(45, len(dataset)):
        print(f"Sample index: {i}")
        sample = dataset[i]
        pos = sample.pos.numpy()
        grasps_dict = sample.y
        pair_scores = sample.pair_scores.numpy()
        mean = sample.sample_info["mean"]
        show_obj_mesh(sample.sample_info, view_params, save=save_image)
        show_grasps_of_edges(pos, grasps_dict[0], pair_scores, dataset.triu_indices, args, sample_info=sample.sample_info,
                        mean=mean, view_params=view_params, save=save_image)
        show_pair_edges(pos, pair_scores, dataset.triu_indices,
                    view_params=view_params, save=save_image)
    

 
