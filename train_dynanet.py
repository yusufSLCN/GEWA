import torch
import torch.nn as nn
import torch_geometric.transforms as T
import wandb
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.transforms import RandomJitter, Compose
import argparse
from tqdm import tqdm
from gewa_dataset import GewaDataset
from DynANet import DynANet
from create_gewa_dataset import save_split_samples
from metrics import check_batch_grasp_success, count_correct_approach_scores
import os
import numpy as np
import torch.optim as optim

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-bs','--batch_size', type=int, default=64)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-d', '--device', type=str, default='cuda')
parser.add_argument('-nw', '--num_workers', type=int, default=0)
parser.add_argument('-nm', '--num_mesh', type=int, default=10)
parser.add_argument('-dd', '--data_dir', type=str, default='../data')
parser.add_argument('-na', '--no_augment', dest='augment', action='store_false')
parser.add_argument('-n', '--notes', type=str, default='')
parser.add_argument('-di', '--device_id', type=int, default=0)
parser.add_argument('-mg', '--multi_gpu', dest='multi_gpu', action='store_true')
parser.add_argument('-cr', '--crop_radius', type=float, default=-1)
parser.add_argument('-gd','--grasp_dim', type=int, default=16)
parser.add_argument('-gs', '--grasp_samples', type=int, default=500)

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
train_dirs, val_dirs = save_split_samples(args.data_dir, num_mesh=args.num_mesh)
train_dataset = GewaDataset(train_dirs)
val_dataset = GewaDataset(val_dirs)
                   
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
config.dataset = train_dataset.__class__.__name__
if args.augment:
    config.transform = transfom_params

config.normalize = train_dataset.normalize_points
config.crop_radius = args.crop_radius
config.grasp_dim = args.grasp_dim
config.grasp_samples = args.grasp_samples

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
model = DynANet(grasp_dim=args.grasp_dim, num_grasp_sample=args.grasp_samples).to(device)

config.model_name = model.__class__.__name__

multi_gpu = False
# If we have multiple GPUs, parallelize the model
if torch.cuda.device_count() > 1 and args.multi_gpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    config.used_gpu_count = torch.cuda.device_count()
    model.multi_gpu = True
    multi_gpu = True
    model = DataParallel(model)
    train_data_loader = DataListLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.num_workers)
    val_data_loader = DataListLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.num_workers)

else:
    train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.num_workers)
    val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.num_workers)

# Define the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000], gamma=0.5)

classification_criterion = nn.BCELoss()
mse_loss = nn.MSELoss()


def calculate_loss(approach_score_pred, grasp_pred, approach_score_gt, grasp_target, approach_points):
    approach_score_gt = (approach_score_gt > 0).float().to(approach_score_pred.device)
    grasp_target = grasp_target.to(grasp_pred.device)
    # print(approach_score_gt.shape, grasp_target.shape)
    approach_loss = classification_criterion(approach_score_pred, approach_score_gt)

    gripper_height = torch.tensor([0, 0, 1.12169998e-01, 1]).to(grasp_pred.device)
    grasp_pred_mat = grasp_pred.reshape(-1, 4, 4)
    grasp_tip = torch.matmul(grasp_pred_mat, gripper_height)[:, :3]
    tip_loss = mse_loss(grasp_tip, approach_points)

    # grasp_target = grasp_target.reshape(-1, 4, 4)
    # target_tip = torch.matmul(grasp_target, gripper_height)[:, :3]
    # dist = (target_tip - approach_points).pow(2).sum(1).sqrt()
    # dist_mask = dist < 0.02
    # grasp_pred = grasp_pred[dist_mask].reshape(-1, 16)
    # grasp_target = grasp_target[dist_mask].reshape(-1, 16)
    grasp_loss = mse_loss(grasp_pred, grasp_target)

    
    return approach_loss, grasp_loss, tip_loss


print(f"Train data size: {len(train_dataset)}")
print(f"Val data size: {len(val_dataset)}")

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    total_grasp_loss = 0
    total_approach_loss = 0
    total_tip_loss = 0
    train_grasp_success = 0
    train_approach_accuracy = 0
    for i, data in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch}/{num_epochs}"):
        optimizer.zero_grad()

        if not multi_gpu:
            data = data.to(device)
            approach_gt = data.approach_scores
        else:
            approach_gt = torch.cat([s.approach_scores for s in data])
        approach_score_pred, grasp_pred, approach_points, grasp_gt = model(data)

        approach_loss, grasp_loss, tip_loss = calculate_loss(approach_score_pred, grasp_pred, approach_gt, grasp_gt, approach_points)

        if multi_gpu:
            grasp_loss = grasp_loss.mean()
            approach_loss = approach_loss.mean()
            tip_loss = tip_loss.mean()

        loss = approach_loss + 50 * tip_loss + grasp_loss
        loss.backward()
        # # Update the weights
        optimizer.step()
        total_loss += loss.item()
        total_grasp_loss += grasp_loss.item()
        total_approach_loss += approach_loss.item()
        total_tip_loss += tip_loss.item()

        if epoch % 20 == 0:
            with torch.no_grad():
                # Calculate the grasp success rate
                pred = grasp_pred.cpu().detach().reshape(-1, 4, 4).numpy()
                gt = grasp_gt.cpu().detach().reshape(-1, 4, 4).numpy()
                train_grasp_success += check_batch_grasp_success(pred, gt,  0.05, np.deg2rad(45)) / len(pred)

                # Calculate the approach accuracy
                if multi_gpu:
                    approach_scores_gt = torch.cat([s.approach_scores for s in data], dim=0).to(approach_score_pred.device)
                else:
                    approach_scores_gt = data.approach_scores

                train_approach_accuracy += count_correct_approach_scores(approach_score_pred, approach_scores_gt)
    scheduler.step()
    if epoch % 20 == 0:
        train_success_rate = train_grasp_success / len(train_data_loader)
        wandb.log({"Train Grasp Success Rate": train_success_rate}, step=epoch)
        train_approach_accuracy = train_approach_accuracy / (len(train_dataset) * 1000)
        wandb.log({"Train Approach Accuracy": train_approach_accuracy}, step=epoch)
    average_loss = total_loss / len(train_data_loader)
    average_grasp_loss = total_grasp_loss / len(train_data_loader)
    average_approach_loss = total_approach_loss / len(train_data_loader)
    average_tip_loss = total_tip_loss / len(train_data_loader)

    # wandb.log({"Train Loss": average_loss}, step=epoch)
    wandb.log({"Train Loss": average_loss, "Train Grasp Loss": average_grasp_loss, "Train Approach Loss": average_approach_loss,
              "Train Tip Loss": average_tip_loss}, step=epoch)
    pred = grasp_pred[0].cpu().detach().numpy()
    gt = grasp_gt[0].cpu().detach().numpy()
    #log the gt and pred gripper pose array with wandb as an example
    wandb.log({"GT Grasp Pose": gt, "Pred Grasp Pose": pred}, step=epoch)

    # Validation loop
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        total_val_grasp_loss = 0
        total_val_approach_loss = 0
        total_val_tip_loss = 0
        valid_grasp_success = 0
        valid_approach_accuracy = 0

        for i, val_data in tqdm(enumerate(val_data_loader), total=len(val_data_loader), desc=f"Valid"):
            # vertices, grasp_gt, batch_idx, querry_point = prepare_samples(device, samples)
            if not multi_gpu:
                val_data = val_data.to(device)
                val_approach_gt = val_data.approach_scores
            else:
                val_approach_gt = torch.cat([s.approach_scores for s in val_data])

            approach_score_pred, val_grasp_pred, approach_points, val_grasp_gt = model(val_data)
            val_approach_loss, val_grasp_loss, val_tip_loss = calculate_loss(approach_score_pred, val_grasp_pred, val_approach_gt, val_grasp_gt, approach_points)
            
            if multi_gpu:
                val_grasp_loss = val_grasp_loss.mean()
                val_approach_loss = val_approach_loss.mean()
            val_loss = val_approach_loss + val_grasp_loss

            if epoch % 20 == 0:
                # Calculate the grasp success rate
                pred = val_grasp_pred.cpu().detach().reshape(-1, 4, 4).numpy()
                gt = val_grasp_gt.cpu().detach().reshape(-1, 4, 4).numpy()
                valid_grasp_success += check_batch_grasp_success(pred, gt,  0.05, np.deg2rad(45)) / len(pred)
                # Calculate the approach accuracy
                if multi_gpu:
                    approach_scores_gt = torch.cat([s.approach_scores for s in val_data], dim=0).to(approach_score_pred.device)
                else:
                    approach_scores_gt = val_data.approach_scores
                valid_approach_accuracy += count_correct_approach_scores(approach_score_pred, approach_scores_gt)

            total_val_loss += val_loss.item()
            total_val_grasp_loss += val_grasp_loss.item()
            total_val_approach_loss += val_approach_loss.item()
            total_val_tip_loss += val_tip_loss.item()

        average_val_loss = total_val_loss / len(val_data_loader)
        average_val_grasp_loss = total_val_grasp_loss / len(val_data_loader)
        average_val_approach_loss = total_val_approach_loss / len(val_data_loader)
        average_val_tip_loss = total_val_tip_loss / len(val_data_loader)
        if epoch % 20 == 0:
            grasp_success_rate = valid_grasp_success / len(val_data_loader)
            wandb.log({"Valid Grasp Success Rate": grasp_success_rate}, step=epoch)
            approach_accuracy = valid_approach_accuracy / (len(val_dataset) * 1000)
            wandb.log({"Valid Approach Accuracy": approach_accuracy}, step=epoch)

            # Save the model if the validation loss is low
            if grasp_success_rate > 0.001:
                model_name = f"{config.model_name}_nm_{args.num_mesh}__bs_{args.batch_size}"
                model_folder = f"models/{model_name}"
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)

                model_file = f"{model_name}_epoch_{epoch}.pth"
                model_path = os.path.join(model_folder, model_file)
                torch.save(model.state_dict(), model_path)
                artifact = wandb.Artifact(model_file, type='model')
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)

    wandb.log({"Val Loss": average_val_loss, "Val Grasp Loss": average_val_grasp_loss, "Val Approach Loss": average_val_approach_loss,
              "Val Tip Loss":average_val_tip_loss}, step=epoch)
    print(f"Train Grasp Loss: {average_grasp_loss:.4f} - Train Approach Loss: {average_approach_loss:.4f} \nVal Grasp Loss: {average_val_grasp_loss:4f} - Val Approach Loss {average_val_approach_loss:.4f}")

# Finish wandb run
wandb.finish()