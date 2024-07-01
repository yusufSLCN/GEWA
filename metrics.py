import numpy as np
import h5py

def is_grasp_success(grasp, target, trans_thresh, rotat_thresh):
    trans_diff = np.linalg.norm(grasp[:3, 3] - target[:3, 3])
    rotat_diff = np.arccos((np.trace(np.dot(grasp[:3, :3].T, target[:3, :3])) - 1) / 2)
    print(trans_diff, rotat_diff)
    if rotat_diff < rotat_thresh and trans_diff < trans_thresh:
        return True
    else:
        return False

def check_grasp_success(grasp, target_file_path, trans_thresh, rotat_thresh, aug_matrix=None):
    data = h5py.File(target_file_path, "r")
    T = np.array(data["grasps/transforms"])
    success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    success_targets = T[np.where(success > 0)]
    if aug_matrix is not None:
        success_targets = np.dot(aug_matrix, success_targets.T).T
    for target in success_targets:
        if is_grasp_success(grasp, target, trans_thresh, rotat_thresh):
            return True
    
    return False
    
def calculate_success_rate(grasps, targets, trans_thresh, rotat_thresh):
    success_count = 0
    for grasp, target in zip(grasps, targets):
        if is_grasp_success(grasp, target, trans_thresh, rotat_thresh):
            success_count += 1
    success_rate = success_count / len(grasps)
    return success_rate

if __name__ == "__main__":
    num_grasps = 1000
    #generate valid random grasps 4x4 matrices
    grasps = np.array([np.eye(4) for i in range(num_grasps)])
    targets = np.array([np.eye(4) for i in range(num_grasps)])
    # scan the rotation from 0 to 180 degrees
    for i in range(num_grasps):
        angle = np.random.uniform(0, np.pi/2)
        r = np.array([[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
        grasps[i, :3, :3] = r


    sr = calculate_success_rate(grasps, targets, 3, np.deg2rad(30))
    print(f"Succes rate: {sr}")