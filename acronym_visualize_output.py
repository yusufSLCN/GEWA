import trimesh
from acronym_utils import create_gripper_marker
from acronym_dataset import AcronymDataset
import numpy as np

gripper = create_gripper_marker()
sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
scene = trimesh.Scene(sphare + gripper)  #Add the ball to the scene

train_dataset = AcronymDataset('sample_dirs/train_success_simplified_acronym_samples.npy')

for i in range(len(train_dataset)):
    sample_idx = i
    sample = train_dataset[sample_idx]
    vertices = sample[0].numpy().astype(np.float64)
    sample_info = sample[2]
    print(f"sample name {sample_info['simplified_model_path']}")

    point_grasp_save_path = sample_info['point_grasp_save_path']
    point_grasp_dict = np.load(point_grasp_save_path, allow_pickle=True).item()
    print(f"Num keys in point grasp dict: {len(point_grasp_dict.keys())}")
    num_grasps = point_grasp_dict[tuple(np.round(vertices[0], 4))][0][0].shape[0]
    if  num_grasps < 5:
        print(f"Limited grasps for this point. {num_grasps} grasps found.")
        break


obj_points = trimesh.points.PointCloud(vertices)
scene.add_geometry(obj_points)

def visualize_random_best_grasps(scene, vertices, point_grasp_dict):
    random_indexes = np.random.choice(len(vertices), 5, replace=False)
    grasp_points = vertices[random_indexes]
    for point in grasp_points:
        sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
        #add color to the sphare
        sphare.visual.face_colors = [0, 255, 0, 255]
        rounded_point = tuple(np.round(point, 4))
        closest_grasp, success = point_grasp_dict[rounded_point][0]
        sphare.apply_translation(point)
        new_gripper = create_gripper_marker()
        point_gripper = new_gripper.apply_transform(closest_grasp)
        scene.add_geometry(sphare)
        scene.add_geometry(point_gripper)
    scene.show()

def visualize_grasps_of_point(scene, point, point_grasp_dict):
    sphare = trimesh.creation.icosphere(subdivisions=4, radius=0.01)
    sphare.visual.face_colors = [0, 255, 0, 255]
    sphare.apply_translation(point)
    scene.add_geometry(sphare)
    rounded_point = tuple(np.round(point, 4))
    for closest_grasp, success in  point_grasp_dict[rounded_point]:
        new_gripper = create_gripper_marker()
        point_gripper = new_gripper.apply_transform(closest_grasp)
        scene.add_geometry(point_gripper)
    scene.show()

# visualize_random_best_grasps(scene, vertices, point_grasp_dict)
visualize_grasps_of_point(scene, vertices[0], point_grasp_dict)
