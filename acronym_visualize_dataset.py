import trimesh
from acronym_utils import create_gripper_marker
from torch_geometric.transforms import RandomJitter, Compose
import numpy as np
import open3d as o3d


def create_scene_with_reference(vertices=None):
    sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
    scene = trimesh.Scene(sphare)  #Add the ball to the scene

    if  vertices is not None:
        obj_points = trimesh.points.PointCloud(vertices)
        scene.add_geometry(obj_points)
    return scene

def visualize_random_best_grasps(vertices, point_grasp_dict, aug_matrix):
    scene = create_scene_with_reference(vertices)
    point_keys = list(point_grasp_dict.keys())
    random_indexes = np.random.choice(len(point_keys), 5, replace=False)
    print(f"Random indexes {random_indexes}")
    for i in random_indexes:
        sphare_query = trimesh.creation.icosphere(subdivisions=4, radius=0.003)
        sphare_contact1 = trimesh.creation.icosphere(subdivisions=4, radius=0.003)
        sphare_contact2 = trimesh.creation.icosphere(subdivisions=4, radius=0.003)

        #add color to the sphare
        sphare_query.visual.face_colors = [0, 255, 0, 255]
        sphare_contact1.visual.face_colors = [255, 0, 255, 255]
        sphare_contact2.visual.face_colors = [255, 0, 255, 255]
        point_key = point_keys[i]
        closest_grasp, success, contact1, contact2 = point_grasp_dict[point_key][0]

        #translation point
        point_key = aug_matrix @ np.array(point_key + (1,))
        point_key = point_key[:3]
        sphare_query.apply_translation(point_key)

        #contact points
        contact1 = aug_matrix @ np.append(contact1[0], 1)
        contact2 = aug_matrix @ np.append(contact2[0], 1)
        contact1 = contact1[:3]
        contact2 = contact2[:3]
        sphare_contact1.apply_translation(contact1)
        sphare_contact2.apply_translation(contact2)

        #gripper
        new_gripper = create_gripper_marker()
        closest_grasp = aug_matrix @ closest_grasp
        closest_grasp[:3, 3] = closest_grasp[:3, 3]
        point_gripper = new_gripper.apply_transform(closest_grasp)
        scene.add_geometry(sphare_query)
        scene.add_geometry(sphare_contact1)
        scene.add_geometry(sphare_contact2)
        scene.add_geometry(point_gripper)
    scene.show()

def visualize_grasps_of_point(vertices, point_idx, point_key, point_grasp_dict, aug_matrix):
    scene = create_scene_with_reference(vertices)
    sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.004)
    # contact1_point = trimesh.creation.icosphere(subdivisions=4, radius=0.004)
    # contact2_point = trimesh.creation.icosphere(subdivisions=4, radius=0.004)
    # contact1_point.visual.face_colors = [255, 0, 255, 255]
    # contact2_point.visual.face_colors = [255, 0, 255, 255]
    sphare.visual.face_colors = [0, 255, 0, 255]
    point = vertices[point_idx]
    sphare.apply_translation(point)
    scene.add_geometry(sphare)
    point_key = tuple(np.round(point_key, 3))
    for closest_grasp, success, contact1, contact2 in  point_grasp_dict[point_key]:
        new_gripper = create_gripper_marker()
        closest_grasp = aug_matrix @ closest_grasp
        # contact1 = aug_matrix @ np.append(contact1[0], 1)
        # contact2 = aug_matrix @ np.append(contact2[0], 1)
        # contact1 = contact1[:3]
        # contact2 = contact2[:3]
        point_gripper = new_gripper.apply_transform(closest_grasp)
        # contact1_point.apply_translation(contact1)
        # contact2_point.apply_translation(contact2)
        scene.add_geometry(point_gripper)
        # scene.add_geometry(contact1_point)
        # scene.add_geometry(contact2_point)
    scene.show()

def visualize_grasp(vertices, grasp, query_point_idx=None):
    scene = create_scene_with_reference(vertices)
    new_gripper = create_gripper_marker()
    point_gripper = new_gripper.apply_transform(grasp)
    scene.add_geometry(point_gripper)

    if query_point_idx is not None:
        sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.005)
        sphare.visual.face_colors = [0, 255, 0, 255]
        query_point = vertices[query_point_idx]
        sphare.apply_translation(query_point)
        scene.add_geometry(sphare)
    scene.show()


def visualize_grasps(vertices, grasps, query_point_idxs=None, contact_points_idx=None, cylinder_edges=None, show_tip=False):
    scene = create_scene_with_reference(vertices)

    if query_point_idxs is None:
        query_point_idxs = [None]*len(grasps)

    if cylinder_edges is not None:
        for cylinder_edge in cylinder_edges:
            edge_sphare1 = trimesh.creation.icosphere(subdivisions=4, radius=0.003)
            edge_sphare1.visual.face_colors = [0, 0, 0, 255]
            edge_sphare1.apply_translation(cylinder_edge[1])
            scene.add_geometry(edge_sphare1)

            edge_sphare2 = trimesh.creation.icosphere(subdivisions=4, radius=0.003)
            edge_sphare2.visual.face_colors = [0, 0, 0, 255]
            edge_sphare2.apply_translation(cylinder_edge[3])
            scene.add_geometry(edge_sphare2)

            
    zipped = zip(grasps, query_point_idxs)    
    for grasp, query_point_idx in zipped:
        new_gripper = create_gripper_marker()
        point_gripper = new_gripper.apply_transform(grasp)
        scene.add_geometry(point_gripper)

        if show_tip:
            gripper_tip_vector = np.array([0, 0, 1.12169998e-01, 1])
            grasp_tip_pos = np.matmul(grasp, gripper_tip_vector)[:3]
            tip = trimesh.creation.icosphere(subdivisions=4, radius=0.003)
            tip.visual.face_colors = [0, 0, 255, 255]
            tip.apply_translation(grasp_tip_pos)
            scene.add_geometry(tip)

        if query_point_idx is not None:
            sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.003)
            sphare.visual.face_colors = [0, 255, 0, 255]
            query_point = vertices[query_point_idx]
            sphare.apply_translation(query_point)
            scene.add_geometry(sphare)


    if contact_points_idx is not None:
        for contact_pair in contact_points_idx:
            contact1_sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.003)
            contact1_sphare.visual.face_colors = [136, 0, 255, 255]
            contact_point1 = vertices[contact_pair[0]]
            contact1_sphare.apply_translation(contact_point1)
            scene.add_geometry(contact1_sphare)

            contact2_sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.003)
            contact2_sphare.visual.face_colors = [51, 255, 204, 255]
            contact_point2 = vertices[contact_pair[1]]
            contact2_sphare.apply_translation(contact_point2)
            scene.add_geometry(contact2_sphare)

    scene.show()

def visualize_gt_and_pred_gasp(vertices, gt, pred, query_point, approach_scores=None):
    if approach_scores is not None:
        sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
        scene = trimesh.Scene(sphare)  #Add the ball to the scene
        import matplotlib.pyplot as plt
        # Choose a colormap
        colormap = plt.cm.viridis
        # approach_prob = (approach_scores - np.min(approach_scores))/ (np.max(approach_scores) - np.min(approach_scores))
        colors = colormap(approach_scores).reshape(-1, 4)
        obj_points = trimesh.points.PointCloud(vertices, colors=colors)
        scene.add_geometry(obj_points)
    else:
        scene = create_scene_with_reference(vertices)

    gt_gripper = create_gripper_marker(color=[0, 255, 0, 255])
    gt_gripper = gt_gripper.apply_transform(gt)
    scene.add_geometry(gt_gripper)
    print(f"gt pose {gt}")

    if query_point is not None:
        sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
        sphare.visual.face_colors = [0, 255, 0, 255]
        sphare.apply_translation(query_point)
        scene.add_geometry(sphare)

        
    pred_gripper = create_gripper_marker(color=[255, 0, 0, 255])
    pred_gripper = pred_gripper.apply_transform(pred)
    scene.add_geometry(pred_gripper)
    print(f"pred pose {pred}")
    scene.show()

def visualize_gt_and_pred_gasps(vertices, gt, pred, query_point, approach_scores=None, num_grasps_of_approach_points=None):
    if approach_scores is not None:
        sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
        scene = trimesh.Scene(sphare)  #Add the ball to the scene
        import matplotlib.pyplot as plt
        # Choose a colormap
        colormap = plt.cm.viridis
        # approach_prob = (approach_scores - np.min(approach_scores))/ (np.max(approach_scores) - np.min(approach_scores))
        colors = colormap(approach_scores).reshape(-1, 4)
        obj_points = trimesh.points.PointCloud(vertices, colors=colors)
        scene.add_geometry(obj_points)
    else:
        scene = create_scene_with_reference(vertices)

    for i in range(len(gt)):

        is_valid_approach_point = True
        if query_point is not None:
            sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
            is_valid_approach_point = num_grasps_of_approach_points[i] > 0
            sphare.visual.face_colors = [0, 255, 0, 255] if is_valid_approach_point else [255, 0, 0, 255]
            sphare.apply_translation(query_point[i])
            scene.add_geometry(sphare)

        if is_valid_approach_point:
            gt_gripper = create_gripper_marker(color=[0, 255, 0, 255])
            gt_gripper = gt_gripper.apply_transform(gt[i])
            scene.add_geometry(gt_gripper)

            
        pred_gripper = create_gripper_marker(color=[255, 0, 0, 255])
        pred_gripper = pred_gripper.apply_transform(pred[i])
        scene.add_geometry(pred_gripper)
        # print(f"pred pose {pred}")
    scene.show()

def visualize_approach_points(vertices, approach_points):
    sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
    scene = trimesh.Scene(sphare)  #Add the ball to the scene

    # colors = np.ones((len(vertices), 4), dtype=np.uint) *255
    # valid_approaches_idx = approach_points > 0
    # colors[valid_approaches] = [0, 255, 0, 255]
    import matplotlib.pyplot as plt
    # Choose a colormap
    colormap = plt.cm.viridis
    # approach_prob = (approach_points - np.min(approach_points))/ (np.max(approach_points) - np.min(approach_points))
    approach_prob = (approach_points > 0) * 1.0
    colors = colormap(approach_prob).reshape(-1, 4)
    obj_points = trimesh.points.PointCloud(vertices, colors=colors)
    scene.add_geometry(obj_points)
    scene.show()

def create_and_show_point_cloud(model_file_name:str):
        mesh_data = o3d.io.read_triangle_mesh(model_file_name)
        point_cloud = mesh_data.sample_points_poisson_disk(1000)
        # point_cloud = mesh_data.sample_points_uniformly(1000)
        point_cloud = np.asarray(point_cloud.points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd])

def visualize_point_cloud(point_cloud: np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.visualization.draw_geometries([pcd])


def visualize_sample(model_path, samples):
    dataset = AcronymDataset(samples)

    sample_found = False
    for i in range(len(dataset)):
        sample = dataset[i]
        # print(sample.sample_info['simplified_model_path'])
        if sample.sample_info['simplified_model_path'] == model_path:
            vertices = sample.pos.numpy().astype(np.float32)
            grasp = sample.y.numpy().astype(np.float32).reshape(4, 4)
            grasp_querry_point = sample.sample_info['query_point']
            query_point_idx = sample.sample_info['query_point_idx']
            point_grasp_dict = np.load(sample.sample_info['point_grasp_save_path'], allow_pickle=True).item()
            print("model path: ", sample.sample_info['model_path'])
            visualize_grasp(vertices, grasp, query_point_idx)
            sample_found = True
            break
    if not sample_found:
        print("Sample not found")

if __name__ == "__main__":
    from gewa_dataset import GewaDataset
    from create_gewa_dataset import save_split_samples


    train_paths, val_paths = save_split_samples('../data', -1)
    # val_dataset = AcronymDataset(val_paths)

    train_dataset = GewaDataset(train_paths)

    sample_idx = 15
    
    sample = train_dataset[sample_idx]

    print(sample)
    # visualize_random_best_grasps(vertices, point_grasp_dict, aug_matrix)
    grasp_path = sample.sample_info['grasps']
    file_name = grasp_path.split("/")[-1]
    simplified_model_path = f"../data/simplified_obj/{file_name}"
    #change extention to obj
    simplified_model_path = simplified_model_path.replace(".h5", ".obj")
    print(simplified_model_path)
    create_and_show_point_cloud(simplified_model_path)
    # create_and_show_point_cloud("../data/simplified_obj/TissueBox_ac6df890acbf354894bed81c37648d8f_0.015413931634988332.obj")
    # visualize_approach_points(vertices, approach_points)
    # visualize_grasps_of_point(vertices, query_point_idx, grasp_querry_point, point_grasp_dict, aug_matrix)
    # print(grasp)
    #check the orthogonality of the grasp
    # print(f"Check col 1 and 2 {np.dot(grasp[:3, 0], grasp[:3, 1])}")
    # print(np.dot(grasp[:3, 0], grasp[:3, 2]))

    # visualize_grasp(vertices, grasp, query_point_idx)
    # print(val_dataset[0].sample_info["simplified_model_path"])
    # visualize_sample("../data/simplified_obj/TissueBox_ac6df890acbf354894bed81c37648d8f_0.015413931634988332.obj", train_paths)
    # visualize_sample("../data/simplified_obj/Bottle_e593aa021f3fa324530647fc03dd20dc_0.007729925649657224.obj", val_paths)
    # visualize_sample(val_dataset[0].sample_info["simplified_model_path"], val_paths)
