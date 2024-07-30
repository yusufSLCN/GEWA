import numpy as np
import h5py
import torch

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
    
def check_batch_grasp_success(grasp_pred, grasp_gt, trans_thresh, rotat_thresh):
    trans_diff = np.linalg.norm(grasp_pred[:, :3, 3] - grasp_gt[:, :3, 3], axis=1)
    h = (np.trace(np.matmul(grasp_pred[:, :3, :3].transpose(0, 2, 1), grasp_gt[:, :3, :3]), axis1=1, axis2=2) - 1) / 2
    h = np.clip(h, -1, 1)
    rotat_diff = np.arccos(h)
    success = np.logical_and(rotat_diff < rotat_thresh, trans_diff < trans_thresh)
    num_success = np.sum(success)
    return num_success 

def check_batch_grasp_success_all_grasps(grasps, grasp_gt_paths, trans_thresh, rotat_thresh, means=None):
    success_count = 0
    # print(len(grasps), len(grasp_gt_paths))
    pred_per_model = len(grasps) // len(grasp_gt_paths)
    # print(pred_per_model)
    if means is not None:
        means = means.detach().numpy().reshape(-1, 3)

    for i in range(len(grasp_gt_paths)):
        target_file_path = grasp_gt_paths[i]
        data = h5py.File(target_file_path, "r")
        T = np.array(data["grasps/transforms"])
        success = np.array(data["grasps/qualities/flex/object_in_gripper"])
        success_targets = T[np.where(success > 0)]
        if means is not None:
            mean = means[i]
            success_targets[:, :3, 3] -= mean

        for grasp in grasps[i * pred_per_model: (i + 1) * pred_per_model]:
            # print(grasp.shape)
            if check_grasp_success_all_grasps(grasp, success_targets, trans_thresh, rotat_thresh):
                success_count += 1
    return success_count

def check_grasp_success_all_grasps(grasp, success_targets, trans_thresh, rotat_thresh):

    if grasp.ndim == 2:
        grasp = grasp.reshape(-1, 4, 4)
    repeated_pred = np.repeat(grasp, len(success_targets), axis=0)
    if check_batch_grasp_success(repeated_pred, success_targets, trans_thresh, rotat_thresh) > 0:
            return True
    
    return False


def check_grasp_success_from_dict(grasp, sample_info, trans_thresh, rotat_thresh):
    point_grasp_save_path = sample_info['point_grasp_save_path']
    point_grasp_dict = np.load(point_grasp_save_path, allow_pickle=True).item()
    point = sample_info['query_point']
    query_point_key = tuple(np.round(point, 3))
    targets = np.array([grasp[0] for grasp in point_grasp_dict[query_point_key]])
    aug_matrix = sample_info['aug_matrix']
    if aug_matrix is not None:
        targets_T = np.transpose(targets, (2, 1, 0))
        targets = np.matmul(aug_matrix, targets_T)
        targets = np.transpose(targets, (2, 1, 0))
        # print(targets.shape)

    for target in targets:
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

def count_correct_approach_scores(approach_pred, approach_gt):
    approach_gt = (approach_gt > 0).float()
    approach_pred = (approach_pred > 0.5).float()
    num_correct = torch.sum(approach_pred == approach_gt).item()
    # acc = num_correct / len(approach_pred)
    return num_correct

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
    from gewa_dataset import GewaDataset
    from create_gewa_dataset import save_split_samples
    from torch_geometric.loader import DataLoader
    # from acronym_dataset import RandomRotationTransform
    import time


    train_paths, val_paths = save_split_samples('../data', -1)
    rotation_range = [-180, 180]

    # transform = RandomRotationTransform(rotation_range)
    train_dataset = GewaDataset(val_paths, normalize_points=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    total_success = 0
    approach_acc = 0

    start_t = time.time()
    num_points = 500
    for i, data in enumerate(train_loader):
        print(f"Batch: {i}/{len(train_loader)}")
        target = data.y.reshape(-1, 4, 4).detach().numpy()
        # success = check_batch_grasp_success(target, target,  0.03, np.deg2rad(30))
        grasp_gt_paths = data.sample_info['grasps']
        # print(target.shape, len(grasp_gt_paths))
        target = [target[i * 1000 : i * 1000 + num_points] for i in range(len(target)//1000)]
        target = np.concatenate(target, axis=0)
        success = check_batch_grasp_success_all_grasps(target, grasp_gt_paths, 0.03, np.deg2rad(30), data.sample_info['mean'])
        # success = check_batch_grasp_success(target, target, 0.03, np.deg2rad(30))
        total_success += success

        # approach_acc += count_correct_approach_scores(data.approach_scores, data.approach_scores)

    print(f"Time: {time.time() - start_t}")


    print(f"Approach accuracy: {approach_acc / (len(train_dataset) * 1000)}")
    print(f"Total success rate: {total_success / (len(train_dataset) * num_points)}")