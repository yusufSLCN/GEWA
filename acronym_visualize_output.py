import trimesh
from acronym_utils import create_gripper_marker
from acronym_dataset import AcronymDataset
import numpy as np
from tqdm import tqdm





def create_scene_with_reference(vertices=None):
    gripper = create_gripper_marker()
    sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
    scene = trimesh.Scene(sphare + gripper)  #Add the ball to the scene

    if  vertices is not None:
        obj_points = trimesh.points.PointCloud(vertices)
        scene.add_geometry(obj_points)
    return scene

def visualize_random_best_grasps(vertices, point_grasp_dict):
    scene = create_scene_with_reference(vertices)
    random_indexes = np.random.choice(len(vertices), 5, replace=False)
    grasp_points = vertices[random_indexes]
    for point in grasp_points:
        sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
        #add color to the sphare
        sphare.visual.face_colors = [0, 255, 0, 255]
        rounded_point = tuple(np.round(point, 3))
        closest_grasp, success = point_grasp_dict[rounded_point][0]
        sphare.apply_translation(point)
        new_gripper = create_gripper_marker()
        point_gripper = new_gripper.apply_transform(closest_grasp)
        scene.add_geometry(sphare)
        scene.add_geometry(point_gripper)
    scene.show()

def visualize_grasps_of_point(vertices, point, point_grasp_dict):
    scene = create_scene_with_reference(vertices)
    sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
    sphare.visual.face_colors = [0, 255, 0, 255]
    sphare.apply_translation(point)
    scene.add_geometry(sphare)
    rounded_point = tuple(np.round(point, 3))
    for closest_grasp, success in  point_grasp_dict[rounded_point]:
        new_gripper = create_gripper_marker()
        point_gripper = new_gripper.apply_transform(closest_grasp)
        scene.add_geometry(point_gripper)
    scene.show()

def visualize_grasp(vertices, grasp, query_point):
    scene = create_scene_with_reference(vertices)
    new_gripper = create_gripper_marker()
    point_gripper = new_gripper.apply_transform(grasp)
    scene.add_geometry(point_gripper)
    sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
    sphare.visual.face_colors = [0, 255, 0, 255]
    sphare.apply_translation(query_point)
    scene.add_geometry(sphare)
    scene.show()

def visualize_gt_and_pred_gasp(vertices, gt, pred, query_point):
    scene = create_scene_with_reference(vertices)
    gt_gripper = create_gripper_marker(color=[0, 255, 0, 255])
    gt_gripper = gt_gripper.apply_transform(gt)
    scene.add_geometry(gt_gripper)
    print(f"gt pose {gt}")

    sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.005)
    sphare.visual.face_colors = [0, 255, 0, 255]
    sphare.apply_translation(query_point)
    scene.add_geometry(sphare)

    pred_gripper = create_gripper_marker(color=[255, 0, 0, 255])
    pred_gripper = pred_gripper.apply_transform(pred)
    scene.add_geometry(pred_gripper)
    print(f"pred pose {pred}")
    scene.show()


if __name__ == "__main__":
    scene = create_scene_with_reference()
    # train_dataset = AcronymDataset('sample_dirs/train_success_simplified_acronym_samples.npy')
    rotation_range = (-np.pi/3, np.pi/3)  # full circle range in radians
    translation_range = (-0.3, 0.3)  # translation values range
    transfom_params = {"rotation_range": rotation_range, "translation_range": translation_range}
    train_dataset = AcronymDataset('sample_dirs/train_success_simplified_acronym_meshes.npy', transfom_params)

    # for i in tqdm(range(len(train_dataset))):
    #     sample_idx = i
    #     sample = train_dataset[sample_idx]
    #     vertices = sample[0].numpy().astype(np.float32)
    #     sample_info = sample[2]

    #     # point_grasp_save_path = sample_info['point_grasp_save_path']
    #     # point_grasp_dict = np.load(point_grasp_save_path, allow_pickle=True).item()
    #     # num_grasps = len(point_grasp_dict[tuple(np.round(vertices[0], 4))])
    #     grap_poses = sample[1]
    #     num_grasps = len(grap_poses)
    #     if num_grasps < 5:
    #         print(f"sample name {sample_info['simplified_model_path']}")
    #         # print(f"Num keys in point grasp dict: {len(point_grasp_dict.keys())}")
    #         print(f"Limited grasps for this point. {num_grasps} grasps found.")
    #         obj_points = trimesh.points.PointCloud(vertices)
    #         scene.add_geometry(obj_points)
    #         for grasp_pose, succ in grap_poses:
    #             new_gripper = create_gripper_marker()
    #             point_gripper = new_gripper.apply_transform(grasp_pose)
    #             scene.add_geometry(point_gripper)
    #         scene.show()
            # break

    sample_idx = 0
    sample = train_dataset[sample_idx]
    vertices = sample[0].numpy().astype(np.float32)
    sample_info = sample[2]
    grasp_querry_point = sample_info['query_point']
    grasp_querry_point = grasp_querry_point.numpy().astype(np.float32)
    point_grasp_dict = np.load(sample_info['point_grasp_save_path'], allow_pickle=True).item()
    grasp = sample[1].reshape(4, 4)
    # visualize_random_best_grasps( vertices, point_grasp_dict)
    # visualize_grasps_of_point(vertices, grasp_querry_point, point_grasp_dict)
    # print(grasp)
    #check the orthogonality of the grasp
    # print(f"Check col 1 and 2 {np.dot(grasp[:3, 0], grasp[:3, 1])}")
    # print(np.dot(grasp[:3, 0], grasp[:3, 2]))

    visualize_grasp(vertices, grasp, grasp_querry_point)
