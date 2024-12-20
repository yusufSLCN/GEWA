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

def check_batch_topk_success_rate(grasp_pred, grasp_gt, trans_thresh, rotat_thresh, num_grasps_of_approach_points, approach_scores, top_k=10):
    num_success = 0
    # print(grasp_pred.shape, grasp_gt.shape)
    # print(num_grasps_of_approach_points.shape)
    for mesh_i in range(grasp_pred.shape[0]):
        top_k_idx = np.argsort(approach_scores[mesh_i])[::-1][:top_k]
        for point_i in top_k_idx:
            num_grasp_of_point_i = num_grasps_of_approach_points[mesh_i, point_i]
            if num_grasp_of_point_i == 0:
                continue
            repeated_pred = np.repeat(grasp_pred[mesh_i, point_i], num_grasp_of_point_i, axis=0)
            grasp_target = grasp_gt[mesh_i, point_i, :num_grasp_of_point_i]
            # print(f"{repeated_pred.shape=}, {grasp_target.shape=}")
            trans_diff = np.linalg.norm(repeated_pred[:, :3, 3] - grasp_target[:, :3, 3], axis=1)
            h = (np.trace(np.matmul(repeated_pred[:, :3, :3].transpose(0, 2, 1), grasp_target[:, :3, :3]), axis1=1, axis2=2) - 1) / 2
            h = np.clip(h, -1, 1)
            rotat_diff = np.arccos(h)
            success = np.logical_and(rotat_diff < rotat_thresh, trans_diff < trans_thresh)
            num_good_grasp_preds = np.sum(success)
            if num_good_grasp_preds > 0:
                num_success += 1
    
    success_rate = num_success / (grasp_pred.shape[0] * top_k)
    return success_rate


def check_succces_with_whole_dataset(grasp_pred, grasp_dict, trans_thresh, rotat_thresh):
    num_success = 0
    for i in range(grasp_pred.shape[0]):
        pred = grasp_pred[i]
        for key in grasp_dict.keys():
            grasp_target = grasp_dict[key]
            repeated_pred = np.repeat(pred, len(grasp_target), axis=0)
            if check_batch_grasp_success(repeated_pred, grasp_target, trans_thresh, rotat_thresh) > 0:
                num_success += 1
                break
    
    success_rate = num_success / grasp_pred.shape[0]
    return success_rate

def check_batch_success_with_whole_gewa_dataset(grasp_preds, trans_thresh, rotat_thresh, grasp_gt_paths, point_cloud_means):
    batch_success_rate = 0
    batch_size = grasp_preds.shape[0]
    # print(grasp_preds.shape, len(grasp_gt_paths), len(point_cloud_means))
    for i in range(batch_size):
        batch_success_rate += check_succces_with_whole_gewa_dataset(grasp_preds[i], trans_thresh, rotat_thresh, grasp_gt_paths[i], point_cloud_means[i])

    batch_success_rate = batch_success_rate / batch_size
    return batch_success_rate

def get_binary_success_with_whole_gewa_dataset(grasp_pred, trans_thresh, rotat_thresh, grasp_gt_path, point_cloud_mean):
    grasp_gt_file = h5py.File(grasp_gt_path, "r")
    grasp_poses = np.array(grasp_gt_file["grasps/transforms"])
    grasp_success = np.array(grasp_gt_file["grasps/qualities/flex/object_in_gripper"])

    grasp_gt = grasp_poses[grasp_success > 0]
    grasp_gt[:, :3, 3] -= point_cloud_mean
    binary_success = np.zeros(grasp_pred.shape[0])
    for i in range(grasp_pred.shape[0]):
        pred = grasp_pred[i]
        repeated_pred = np.repeat(pred, len(grasp_gt), axis=0)
        # print(repeated_pred.shape, grasp_gt.shape)
        if check_batch_grasp_success(repeated_pred, grasp_gt, trans_thresh, rotat_thresh) > 0:
            binary_success[i] = 1

    return binary_success

def check_succces_with_whole_gewa_dataset(grasp_pred, trans_thresh, rotat_thresh, grasp_gt_path, point_cloud_mean):
    grasp_gt_file = h5py.File(grasp_gt_path, "r")
    grasp_poses = np.array(grasp_gt_file["grasps/transforms"])
    grasp_success = np.array(grasp_gt_file["grasps/qualities/flex/object_in_gripper"])

    grasp_gt = grasp_poses[grasp_success > 0]
    grasp_gt[:, :3, 3] -= point_cloud_mean
    num_success = 0
    for i in range(grasp_pred.shape[0]):
        pred = grasp_pred[i]
        repeated_pred = np.repeat(pred, len(grasp_gt), axis=0)
        # print(repeated_pred.shape, grasp_gt.shape)
        if check_batch_grasp_success(repeated_pred, grasp_gt, trans_thresh, rotat_thresh) > 0:
            num_success += 1

    success_rate = num_success / grasp_pred.shape[0]
    return success_rate
                                         
                                         

def check_batch_grasp_success_rate_per_point(grasp_pred, grasp_gt, trans_thresh, rotat_thresh, num_grasps_of_approach_points):
    num_success = 0
    num_valid_points = np.sum(num_grasps_of_approach_points > 0)
    if num_valid_points == 0:
        print("No valid points")
        return 0
    # print(grasp_pred.shape, grasp_gt.shape)
    # print(num_grasps_of_approach_points.shape)
    for mesh_i in range(grasp_pred.shape[0]):
        for point_i in range(grasp_pred.shape[1]):
            num_grasp_of_point_i = num_grasps_of_approach_points[mesh_i, point_i]
            if num_grasp_of_point_i == 0:
                continue
            repeated_pred = np.repeat(grasp_pred[mesh_i, point_i], num_grasp_of_point_i, axis=0)
            grasp_target = grasp_gt[mesh_i, point_i, :num_grasp_of_point_i]
            # print(f"{repeated_pred.shape=}, {grasp_target.shape=}")
            trans_diff = np.linalg.norm(repeated_pred[:, :3, 3] - grasp_target[:, :3, 3], axis=1)
            h = (np.trace(np.matmul(repeated_pred[:, :3, :3].transpose(0, 2, 1), grasp_target[:, :3, :3]), axis1=1, axis2=2) - 1) / 2
            h = np.clip(h, -1, 1)
            rotat_diff = np.arccos(h)
            success = np.logical_and(rotat_diff < rotat_thresh, trans_diff < trans_thresh)
            num_good_grasp_preds = np.sum(success)
            if num_good_grasp_preds > 0:
                num_success += 1
    
    success_rate = num_success / num_valid_points
    return success_rate

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
    from dataset.approach_dataset import ApproachDataset
    from dataset.create_approach_dataset import save_contactnet_split_samples
    from torch_geometric.loader import DataLoader
    # from acronym_dataset import RandomRotationTransform
    import time


    train_paths, val_paths = save_contactnet_split_samples('../data', 100)
    rotation_range = [-180, 180]

    # transform = RandomRotationTransform(rotation_range)
    train_dataset = ApproachDataset(val_paths, normalize_points=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=0)
    total_success = 0
    approach_acc = 0

    start_t = time.time()
    num_points = 1000
    for i, data in enumerate(train_loader):
        print(f"Batch: {i}/{len(train_loader)}")
        target = data.y.reshape(-1, 1000, 20, 4, 4).detach().numpy()
        # success = check_batch_grasp_success(target, target,  0.03, np.deg2rad(30))
        # grasp_gt_paths = data.sample_info['grasps']
        # print(target.shape, len(grasp_gt_paths))
        # target = [target[i * 1000 : i * 1000 + num_points] for i in range(len(target)//1000)]
        # target = np.concatenate(target, axis=0)
        pred = np.expand_dims(target[:, :, 0], axis=2)
        # success = check_batch_grasp_success_all_grasps(target, grasp_gt_paths, 0.03, np.deg2rad(30), data.sample_info['mean'])
        num_grasp_of_approach_points = data.num_grasps.detach().numpy()
        num_grasp_of_approach_points = num_grasp_of_approach_points.reshape(-1, 1000)
        success = check_batch_grasp_success_rate_per_point(pred, target, 0.03, np.deg2rad(30), num_grasp_of_approach_points)
        total_success += success

        # approach_acc += count_correct_approach_scores(data.approach_scores, data.approach_scores)

    print(f"Time: {time.time() - start_t}")


    print(f"Approach accuracy: {approach_acc / (len(train_dataset) * 1000)}")
    print(f"Total success rate: {total_success / len(train_loader)}")