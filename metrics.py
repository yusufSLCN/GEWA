import numpy as np
import h5py

def is_grasp_success(grasp, target, trans_thresh, rotat_thresh):
    trans_diff = np.linalg.norm(grasp[:3, 3] - target[:3, 3])
    h = (np.trace(np.matmul(grasp[:3, :3].T, target[:3, :3])) - 1) / 2
    h = np.clip(h, -1, 1)
    rotat_diff = np.arccos(h)
    if rotat_diff < rotat_thresh and trans_diff < trans_thresh:
        # print(trans_diff, rotat_diff)
        return True
    else:
        return False

def check_grasp_success(grasp, target_file_path, trans_thresh, rotat_thresh, aug_matrix=None):
    data = h5py.File(target_file_path, "r")
    T = np.array(data["grasps/transforms"])
    success = np.array(data["grasps/qualities/flex/object_in_gripper"])
    success_targets = T[np.where(success > 0)]
    if aug_matrix is not None:
        success_targets = np.matmul(aug_matrix, success_targets.T).T
        print(success_targets.shape)

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
    # num_grasps = 1000
    # #generate valid random grasps 4x4 matrices
    # grasps = np.array([np.eye(4) for i in range(num_grasps)])
    # targets = np.array([np.eye(4) for i in range(num_grasps)])
    # # scan the rotation from 0 to 180 degrees
    # total_success = 0
    # for i in range(num_grasps):
    #     angle = np.random.uniform(0, np.pi/2)
    #     r = np.array([[np.cos(angle), -np.sin(angle), 0],
    #                   [np.sin(angle), np.cos(angle), 0],
    #                   [0, 0, 1]])
    #     grasps[i, :3, :3] = r
    #     total_success += is_grasp_success(grasps[i], targets[i], 0.03, np.deg2rad(30))

    # print(f"Total success rate: {total_success / num_grasps}")


    # print(f"Succes rate: {sr}")

    from acronym_dataset import AcronymDataset
    from create_dataset_paths import save_split_meshes
    from torch_geometric.loader import DataListLoader
    from acronym_dataset import RandomRotationTransform


    train_paths, val_paths = save_split_meshes('../data', 100)
    rotation_range = [-180, 180]

    transform = RandomRotationTransform(rotation_range)
    train_dataset = AcronymDataset(train_paths, crop_radius=None, transform=transform, normalize_vertices=True)
    train_loader = DataListLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    total_success = 0

    for i, data in enumerate(train_loader):
        for i in range(len(data)):
            sample = data[i]
            target = sample.y.reshape(4, 4)
            aug_matrix = sample.sample_info['aug_matrix']
            # point_grasp_save_path = sample.sample_info['point_grasp_save_path']
            # point_grasp_dict = np.load(point_grasp_save_path, allow_pickle=True).item()
            # target = point_grasp_dict[list(point_grasp_dict.keys())[0]][0][0].reshape(4, 4)
            # target = torch.tensor(target, dtype=torch.float32)

            #load grasps
            grasps_file_name = sample.sample_info['grasps']
            # grasps = h5py.File(grasps_file_name, "r")
            # grasp_poses = np.array(grasps["grasps/transforms"])
            # grasp_success = np.array(grasps["grasps/qualities/flex/object_in_gripper"])
            # grasp_poses = grasp_poses[np.where(grasp_success > 0)]
            # target =  torch.tensor(grasp_poses[0])
            success = check_grasp_success(target, grasps_file_name, 0.03, np.deg2rad(30), aug_matrix=aug_matrix)
            total_success += success

    print(f"Total success rate: {total_success / len(train_dataset)}")