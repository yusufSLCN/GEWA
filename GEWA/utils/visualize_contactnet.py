import numpy as np
import os
import h5py
from utils.visualize_acronym_dataset import visualize_grasps

if __name__ == "__main__":
    results_path = "../contactgraspnet/results"
    files = os.listdir(results_path)
    for file in files:
        results = np.load(os.path.join(results_path, file), allow_pickle=True)
        print(results.keys())
        point_cloud = results['point_cloud']
        pred_grasps_cam = results['pred_grasps_cam']
        grasp_gt_path = results['grasp_gt_path'].item()
        print(point_cloud.shape, pred_grasps_cam.shape, grasp_gt_path)
        #laod gt grasps
        data = h5py.File(grasp_gt_path, "r")
        grasp_poses = np.array(data["grasps/transforms"])
        grasp_success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        grasp_gt = grasp_poses[grasp_success > 0.5]

        display_count = 10
        replace = True if len(pred_grasps_cam) < display_count else False
        selected_indexes = np.random.choice(pred_grasps_cam.shape[0], display_count, replace=replace)
        selected_pred_grasps = pred_grasps_cam[selected_indexes]
        visualize_grasps(point_cloud, selected_pred_grasps)
    
