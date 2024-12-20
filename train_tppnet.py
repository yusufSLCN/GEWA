import torch
import torch.nn as nn
import torch_geometric.transforms as T
import wandb
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.transforms import RandomJitter, Compose
import argparse
from tqdm import tqdm
from dataset.tpp_dataset import TPPDataset
from models.TppNet import TppNet
from models.TppBallNet import TppBallNet
from dataset.create_tpp_dataset import save_contactnet_split_samples
from utils.metrics import check_batch_success_with_whole_gewa_dataset, check_batch_grasp_success_rate_per_point
import os
import numpy as np
import torch.optim as optim
import time
# from sklearn.metrics import recall_score
from torcheval.metrics.functional.classification import binary_accuracy, binary_recall, binary_precision, binary_f1_score

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-bs','--batch_size', type=int, default=2)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-d', '--device', type=str, default='cuda')
parser.add_argument('-nw', '--num_workers', type=int, default=0)
parser.add_argument('-nm', '--num_mesh', type=int, default=10)
parser.add_argument('-dd', '--data_dir', type=str, default='../data')
parser.add_argument('-a', '--augment', dest='augment', action='store_true')
parser.add_argument('-n', '--notes', type=str, default='')
parser.add_argument('-di', '--device_id', type=int, default=0)
parser.add_argument('-mg', '--multi_gpu', dest='multi_gpu', action='store_true')
parser.add_argument('-cr', '--crop_radius', type=float, default=-1)
parser.add_argument('-gd','--grasp_dim', type=int, default=7)
parser.add_argument('-gs', '--grasp_samples', type=int, default=100)
parser.add_argument('-li', '--log_interval', type=int, default=10)
parser.add_argument('-oc', '--only_classifier', action='store_true')
parser.add_argument('-mn', '--model_name', type=str, default='tppnet')
# parser.add_argument('-csplit', '--contactnet_split', action='store_true')
parser.add_argument('-dn', '--dataset_name', type=str, default="tpp_effdict_nomean_wnormals")
args = parser.parse_args()


# Load the datasets with transforms
if args.augment:
    translation_range = 0.001  # translation values range
    # rotation_range = [-180, 180]
    rotation_range = None
    transform_list = [RandomJitter(translation_range)]
    # if rotation_range is not None:
    #     transform_list.append(RandomRotationTransform(rotation_range))
    transform = Compose(transform_list)
    transfom_params = f"Translation Range: {translation_range}, Rotation Range: {rotation_range}"
else:
    transform = None
    transfom_params = None

print("Transform params: ", transfom_params)

# Save the split samples
# train_dirs, val_dirs = save_split_samples(args.data_dir, num_mesh=args.num_mesh, dataset_name=args.dataset_name,
#                                            contactnet_split=args.contactnet_split)

train_dirs, val_dirs = save_contactnet_split_samples(args.data_dir, num_mesh=args.num_mesh, dataset_name=args.dataset_name)
return_grasp_dict = not args.only_classifier
train_dataset = TPPDataset(train_dirs, transform=transform, return_pair_dict=return_grasp_dict, normalize=True)
val_dataset = TPPDataset(val_dirs, return_pair_dict=return_grasp_dict, normalize=True)
                   
# Initialize wandb
wandb.init(project="Grasp", notes=args.notes)

# Log hyperparameters
config = wandb.config
config.learning_rate = args.learning_rate
config.batch_size = args.batch_size
config.epoch = args.epochs
config.num_mesh = args.num_mesh
config.data_dir = args.data_dir
config.num_workers = args.num_workers
config.contactnet_split = True
config.dataset = train_dataset.__class__.__name__
config.train_size = len(train_dataset)
config.val_size = len(val_dataset)
if args.augment:
    config.transform = transfom_params

config.crop_radius = args.crop_radius
config.grasp_dim = args.grasp_dim
config.grasp_samples = args.grasp_samples
config.only_classifier = args.only_classifier
config.dataset_name = args.dataset_name
# Analyze the dataset class stats
num_epochs = args.epochs

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device(f"cuda:{args.device_id}")
else:
    device = torch.device('cpu')

config.device = device
print(device)

# Initialize the model
# model = GraspNet(scene_feat_dim= config.scene_feat_dims).to(device)
# model = GewaNet(scene_feat_dim= config.scene_feat_dims, device=device).to(device)
max_grasp_per_edge = 10
topk = 10
if args.model_name == 'tppnet':
    model = TppNet(grasp_dim=args.grasp_dim, num_grasp_sample=args.grasp_samples,
                max_num_grasps=max_grasp_per_edge, only_classifier=args.only_classifier, normalize=True, topk=topk).to(device)
elif args.model_name == 'tppballnet':
    model = TppBallNet(grasp_dim=args.grasp_dim, num_grasp_sample=args.grasp_samples,
                max_num_grasps=max_grasp_per_edge, only_classifier=args.only_classifier, normalize=True, topk=topk).to(device)

num_pairs = model.num_pairs
config.model_name = model.__class__.__name__
config.topk = topk
wandb.watch(model, log="all")

multi_gpu = False
# If we have multiple GPUs, parallelize the model
if torch.cuda.device_count() > 1 and args.multi_gpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    config.used_gpu_count = torch.cuda.device_count()
    model.multi_gpu = True
    multi_gpu = True
    model = DataParallel(model)
    train_data_loader = DataListLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_data_loader = DataListLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

else:
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs // 2, 3 * args.epochs // 4], gamma=0.5)

# classification_criterion = nn.BCELoss()
# classification_criterion = nn.BCEWithLogitsLoss(pos_weight=500)
tip_mse_loss = nn.MSELoss()
grasp_axis_mse_loss = nn.MSELoss()
grasp_mse_loss = nn.MSELoss(reduction='none')
# grasp_mse_loss = nn.MSELoss()
neg_pos_ratio = 44.8
classification_criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor(0.3 * neg_pos_ratio))


def calculate_loss(grasp_pred, grasp_target, num_valid_grasps, mid_edge_points, mlp_out_ij, binary_pair_scores_gt, grasp_axises=None, num_grasp_samples=50):
    # Calculate the pair loss
    # pos_pair_count = torch.sum(binary_pair_scores_gt)
    # pos_weight = (binary_pair_scores_gt.numel() - pos_pair_count) / pos_pair_count
    # classification_criterion = nn.BCEWithLogitsLoss(pos_weight= 0.1 * pos_weight)
    pair_loss = classification_criterion(mlp_out_ij, binary_pair_scores_gt)

    if args.only_classifier:
        return pair_loss, None, None

    # print(mid_edge_points.shape)
    valid_grasp_mask = num_valid_grasps > 0
    valid_grasp_mask = valid_grasp_mask.flatten()
    # Calculate the tip loss
    gripper_height = torch.tensor([0, 0, 1.12169998e-01, 1]).to(grasp_pred.device)
    grasp_pred_mat = grasp_pred.reshape(-1, 4, 4)
    grasp_tip = torch.matmul(grasp_pred_mat, gripper_height)[:, :3]
    grasp_tip = grasp_tip[valid_grasp_mask]
    mid_edge_points = mid_edge_points[valid_grasp_mask]
    tip_loss = tip_mse_loss(grasp_tip, mid_edge_points)

    #calculate grasp axis loss
    if grasp_axises is not None:
        pred_grasp_axis = grasp_pred_mat[:, :3, 0]
        # pred_grasp_axis = pred_grasp_axis.reshape(-1, 3)
        dot_product = torch.sum(pred_grasp_axis * grasp_axises, dim=-1)
        squared_dot = dot_product ** 2
        grasp_axis_loss = -torch.mean(squared_dot)
    else:
        grasp_axis_loss = torch.tensor([0.0]).to(grasp_pred.device)


    # # Calculate the grasp loss
    grasp_pred = grasp_pred.reshape(-1, num_grasp_samples, 1,  16)
    min_grasp_losses = []
    grasp_target = grasp_target.to(grasp_pred.device)
    grasp_target = grasp_target.reshape(-1, num_grasp_samples, max_grasp_per_edge, 16)

    #just translation
    # grasp_pred = grasp_pred_mat[:, :3, 3]
    # grasp_pred = grasp_pred.reshape(-1, args.grasp_samples, 1,  3)
    # grasp_target = grasp_target[:, :, :, :3, 3]
    # grasp_target = grasp_target.to(grasp_pred.device)

    for i in range(len(grasp_target)):
        sample_grasp_target = grasp_target[i]
        sample_grasp_pred = grasp_pred[i].repeat(1, max_grasp_per_edge, 1)
        grasp_losses = grasp_mse_loss(sample_grasp_pred, sample_grasp_target)
        grasp_losses = grasp_losses.sum(dim=-1)

        #some of the target grasps are just for padding
        #extract the valid grasp losses and take the minimum
        for g_idx in range(num_grasp_samples):
            if num_valid_grasps[i, g_idx] == 0:
                continue
            a_point_loss = grasp_losses[g_idx, :num_valid_grasps[i, g_idx]]
            min_grasp_loss = torch.min(a_point_loss)
            # min_grasp_loss = a_point_loss[0]
            min_grasp_losses.append(min_grasp_loss)

    if len(min_grasp_losses) > 0:
        grasp_loss = sum(min_grasp_losses) / len(min_grasp_losses)
    else:
        grasp_loss = torch.tensor([0.0]).to(grasp_pred.device)
    
    return pair_loss, grasp_loss, tip_loss, grasp_axis_loss


print(f"Train data size: {len(train_dataset)}")
print(f"Val data size: {len(val_dataset)}")
# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    total_grasp_loss = 0
    total_pair_loss = 0
    total_tip_loss = 0
    total_grasp_axis_loss = 0
    train_grasp_success = 0
    train_pair_accuracy = 0
    train_recall = 0
    train_precision = 0
    train_f1 = 0
    train_grasp_success = 0
    for i, data in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch}/{num_epochs}"):
        optimizer.zero_grad()
        t1 = time.time()
        if not multi_gpu:
            # data = data.to(device)
            data.pos = data.pos.to(device)
            data.batch = data.batch.to(device)
            pair_scores_gt = data.pair_scores
        else:
            pair_scores_gt = torch.cat([s.pair_scores for s in data])

        t2 = time.time()
        # print(f"Data prep takes {t2 - t1}s")
        # if args.only_classifier:
        #     pair_classification_pred, mlp_out_ij = model(data)
        # else:
        grasp_pred, selected_edge_idxs, mid_edge_pos, grasp_axises, grasp_target, num_valid_grasps, pair_classification_pred, mlp_out_ij = model(data)

        t3 = time.time()
        # print(f"Model takes {t3 - t2}s")
        pair_scores_gt = pair_scores_gt.reshape(-1, num_pairs)
        binary_pair_scores_gt = (pair_scores_gt > 0).float().to(pair_classification_pred.device)

        pair_loss, grasp_loss, tip_loss, grasp_axis_loss = calculate_loss(grasp_pred, grasp_target, num_valid_grasps, mid_edge_pos,
                                                                           mlp_out_ij, binary_pair_scores_gt, grasp_axises, num_grasp_samples=args.grasp_samples)

        # contrastive_criterion = nn.CrossEntropyLoss()
        # contrastive_loss =  contrastive_criterion(mlp_out_ij, mlp_out_ji)
        t4 = time.time()
        # print(f"Loss takes {t4 - t3}s")
        if multi_gpu:
            pair_loss = pair_loss.mean()
            if not args.only_classifier:
                grasp_loss = grasp_loss.mean()
                tip_loss = tip_loss.mean()
                grasp_axis_loss = grasp_axis_loss.mean()
            # contrastive_loss = contrastive_loss.mean()

        # if args.only_classifier:
        #     loss = pair_loss
        # else:
            # loss = grasp_loss + 500 * tip_loss + pair_loss
        # loss = 200 *grasp_loss + pair_loss + grasp_axis_loss
        loss = grasp_loss + pair_loss + grasp_axis_loss

        loss.backward()
        # # Update the weights
        optimizer.step()
        total_loss += loss.item()
        total_pair_loss += pair_loss.item()
        total_grasp_axis_loss += grasp_axis_loss.item()
        if not args.only_classifier:
            total_grasp_loss += grasp_loss.item()
            total_tip_loss += tip_loss.item()

        if epoch % args.log_interval == 0:
            with torch.no_grad():
                if not args.only_classifier:
                    grasp_pred = grasp_pred.cpu().detach().reshape(-1, args.grasp_samples, 1, 4, 4).numpy()

                    if multi_gpu:
                        grasp_gt_paths = [s.sample_info['grasps'] for s in data]
                        means = [s.sample_info['mean'] for s in data]
                    else:
                        grasp_gt_paths = data.sample_info['grasps']
                        means = data.sample_info['mean']
                    
                    train_grasp_success += check_batch_success_with_whole_gewa_dataset(grasp_pred, 0.03, np.deg2rad(30),grasp_gt_paths, means)

                # Calculate the pair accuracy
                pair_classification_pred = pair_classification_pred.to(binary_pair_scores_gt.device)
                pair_classification_pred = torch.flatten(pair_classification_pred)
                binary_pair_scores_gt = torch.flatten(binary_pair_scores_gt).int()
                train_pair_accuracy += binary_accuracy(pair_classification_pred, binary_pair_scores_gt)
                train_recall += binary_recall(pair_classification_pred, binary_pair_scores_gt)
                train_precision += binary_precision(pair_classification_pred, binary_pair_scores_gt)
                train_f1 += binary_f1_score(pair_classification_pred, binary_pair_scores_gt)

    # average_loss = total_loss / len(train_data_loader)
    average_grasp_loss = total_grasp_loss / len(train_data_loader)
    average_pair_loss = total_pair_loss / len(train_data_loader)
    average_tip_loss = total_tip_loss / len(train_data_loader)
    average_grasp_axis_loss = total_grasp_axis_loss / len(train_data_loader)

    # wandb.log({"Train Loss": average_loss}, step=epoch)
    wandb.log({"Train Pair Loss": average_pair_loss, "Train Tip Loss":average_tip_loss,
                "Train Grasp Loss":average_grasp_loss, "Train Grasp Axis Loss": average_grasp_axis_loss}, step=epoch)

    scheduler.step()
    if epoch % args.log_interval == 0:
        if not args.only_classifier:
            train_success_rate = train_grasp_success / len(train_data_loader)
            wandb.log({"Train Grasp Success Rate": train_success_rate}, step=epoch)

        train_pair_accuracy = train_pair_accuracy / len(train_data_loader)
        train_f1 = train_f1 / len(train_data_loader)
        train_precision = train_precision / len(train_data_loader)
        train_recall = train_recall / len(train_data_loader)
        wandb.log({"Train Recall": train_recall, "Train Accuracy":train_pair_accuracy,
                    "Train Precision":train_precision, "Train_F1":train_f1}, step=epoch)
    # Validation loop
    if epoch % args.log_interval == 0:
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_grasp_loss = 0
            total_val_pair_loss = 0
            total_val_tip_loss = 0
            total_val_grasp_axis_loss = 0
            valid_pair_accuracy = 0
            val_recall = 0
            val_precision = 0
            val_f1 = 0
            val_grasp_success = 0
            for i, val_data in tqdm(enumerate(val_data_loader), total=len(val_data_loader), desc=f"Valid"):
                if not multi_gpu:
                    # val_data = val_data.to(device)
                    val_data.pos = val_data.pos.to(device)
                    val_data.batch = val_data.batch.to(device)
                    val_pair_scores_gt = val_data.pair_scores.to(device)
                else:
                    val_pair_scores_gt = torch.cat([s.pair_scores for s in val_data])

                
                # val_pair_pred, val_pair_dot_product = model(val_data)
                val_grasp_pred, val_selected_edge_idxs, val_mid_edge_pos, val_grasp_axises, val_grasp_target, val_num_valid_grasps, val_pair_pred, val_mlp_out_ij = model(val_data)
                
                val_pair_scores_gt = val_pair_scores_gt.reshape(-1, num_pairs)
                val_binary_pair_scores_gt = (val_pair_scores_gt > 0).float().to(val_pair_pred.device)

                val_pair_loss, val_grasp_loss, val_tip_loss, val_grasp_axis_loss = calculate_loss(val_grasp_pred, val_grasp_target, val_num_valid_grasps,
                                                                            val_mid_edge_pos, val_mlp_out_ij, val_binary_pair_scores_gt, val_grasp_axises, topk)

                if args.only_classifier:
                    val_loss = val_pair_loss
                    if multi_gpu:
                        val_pair_loss = val_pair_loss.mean()
                    total_val_pair_loss += val_pair_loss.item()
                else:
                    if multi_gpu:
                        val_pair_loss = val_pair_loss.mean()
                        val_grasp_loss = val_grasp_loss.mean()
                        val_tip_loss = val_tip_loss.mean()
                        val_grasp_axis_loss = val_grasp_axis_loss.mean()

                    total_val_pair_loss += val_pair_loss.item()
                    total_val_grasp_loss += val_grasp_loss.item()
                    total_val_tip_loss += val_tip_loss.item()
                    total_val_grasp_axis_loss += val_grasp_axis_loss.item()

                
                if not args.only_classifier:
                    val_grasp_pred = val_grasp_pred.cpu().detach().reshape(-1, topk, 1, 4, 4).numpy()
                    # val_grasp_target = val_grasp_target.cpu().detach().reshape(-1, args.grasp_samples, max_grasp_per_edge, 4, 4).numpy()
                    # val_num_valid_grasps = val_num_valid_grasps.cpu().detach().numpy()
                    # val_grasp_success += check_batch_grasp_success_rate_per_point(val_grasp_pred, val_grasp_target, 0.03,
                    #                                                                 np.deg2rad(30), val_num_valid_grasps)
                    if multi_gpu:
                        val_grasp_gt_paths = [s.sample_info['grasps'] for s in val_data]
                        val_means = [s.sample_info['mean'] for s in val_data]
                    else:
                        val_grasp_gt_paths = val_data.sample_info['grasps']
                        val_means = val_data.sample_info['mean']

                    val_grasp_success += check_batch_success_with_whole_gewa_dataset(val_grasp_pred, 0.03, np.deg2rad(30),val_grasp_gt_paths, val_means)


                #sklearn to get other metrics
                val_pair_pred = torch.flatten(val_pair_pred)
                val_binary_pair_scores_gt = torch.flatten(val_binary_pair_scores_gt).int()
                valid_pair_accuracy += binary_accuracy(val_pair_pred, val_binary_pair_scores_gt)
                val_recall += binary_recall(val_pair_pred, val_binary_pair_scores_gt)
                val_precision += binary_precision(val_pair_pred, val_binary_pair_scores_gt)
                val_f1 += binary_f1_score(val_pair_pred, val_binary_pair_scores_gt)


            # average_val_loss = total_val_loss / len(val_data_loader)
            average_val_pair_loss = total_val_pair_loss / len(val_data_loader)
            average_val_grasp_loss = total_val_grasp_loss / len(val_data_loader)
            average_val_tip_loss = total_val_tip_loss / len(val_data_loader)
            average_val_grasp_axis_loss = total_val_grasp_axis_loss / len(val_data_loader)

            val_grasp_success_rate = val_grasp_success / len(val_data_loader)

            wandb.log({"Valid Grasp Success Rate": val_grasp_success_rate}, step=epoch)

            valid_pair_accuracy = valid_pair_accuracy / len(val_data_loader)
            val_recall = val_recall / len(val_data_loader)
            val_precision = val_precision / len(val_data_loader)
            val_f1 = val_f1 / len(val_data_loader)
            wandb.log({"Val Recall": val_recall, "Val Accuracy":valid_pair_accuracy, 
                       "Val Precision":val_precision, "Val_F1":val_f1}, step=epoch)
            print(f"Train Pair Acc: {train_pair_accuracy} - Valid Pair Accuracy: {valid_pair_accuracy}")
            print(f"Train Pair Recall: {train_recall} - Valid Pair Recall: {val_recall}")
            print(f"Train Pair Precision: {train_precision} - Valid Pair Precision: {val_precision}")
            print(f"Train Pair F1: {train_f1} - Valid Pair F1: {val_f1}")
            print(f"Train Grasp Success: {train_success_rate} - Valid Grasp Success: {val_grasp_success_rate}")
            # Save the model if the validation loss is low
            if val_grasp_success_rate > 0.1:
                model_name = f"{config.model_name}_nm_{args.num_mesh}__bs_{args.batch_size}.pth"
                model_folder = f"saved_models/{model_name}"
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)

                model_file = f"{model_name}_epoch_{epoch}_success_{val_grasp_success_rate:.2f}_acc_{valid_pair_accuracy:.2f}_recall_{val_recall:.2f}.pth"
                model_path = os.path.join(model_folder, model_file)
                torch.save(model.state_dict(), model_path)
                artifact = wandb.Artifact(model_file, type='model')
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)

        wandb.log({"Val Pair Loss": average_val_pair_loss, "Val Grasp Loss": average_val_grasp_loss,
                        "Val Tip Loss": average_tip_loss, "Val Grasp Axis Loss": average_grasp_axis_loss}, step=epoch)
        print(f" Val Pair Loss: {average_val_pair_loss:.4f} - Val Grasp Loss: {average_val_grasp_loss:.4f}")
    print(f"Train Pair Loss: {average_pair_loss:.4f} - Train Grasp Loss: {average_grasp_loss:.4f}")

# Finish wandb run
wandb.finish()