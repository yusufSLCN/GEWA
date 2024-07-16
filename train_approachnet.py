import torch
import torch.nn as nn
import torch_geometric.transforms as T
import wandb
from torch_geometric.loader import DataLoader
from torch_geometric.nn import DataParallel
from torch_geometric.transforms import RandomJitter, Compose
import argparse
from tqdm import tqdm
from gewa_dataset import GewaDataset
# from EdgeGraspNet import EdgeGraspNet
from GewaNet import GewaNet
from ApproachNet import ApproachNet
from create_gewa_dataset import save_split_samples
from metrics import check_batch_grasp_success
import os
import numpy as np

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-bs','--batch_size', type=int, default=64)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-d', '--device', type=str, default='cuda')
parser.add_argument('-nw', '--num_workers', type=int, default=0)
parser.add_argument('-nm', '--num_mesh', type=int, default=10)
parser.add_argument('-dd', '--data_dir', type=str, default='../data')
parser.add_argument('-na', '--no_augment', dest='augment', action='store_false')
parser.add_argument('-sfd', '--scene_feat_dims', type=int, default=1024)
parser.add_argument('-n', '--notes', type=str, default='')
parser.add_argument('-di', '--device_id', type=int, default=0)
parser.add_argument('-mg', '--multi_gpu', dest='multi_gpu', action='store_true')
parser.add_argument('-cr', '--crop_radius', type=float, default=-1)
parser.add_argument('-gd','--grasp_dim', type=int, default=9)
parser.add_argument('-gs', '--grasp_samples', type=int, default=1000)
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
train_dataset = GewaDataset(train_dirs, transform=transform)
val_dataset = GewaDataset(val_dirs)
                   
# Initialize wandb
wandb.init(project="GEWA", notes=args.notes)

# Log hyperparameters
config = wandb.config
config.learning_rate = args.learning_rate
config.batch_size = args.batch_size
config.epoch = args.epochs
config.num_mesh = args.num_mesh
config.data_dir = args.data_dir
config.num_workers = args.num_workers
config.dataset = train_dataset.__class__.__name__
config.scene_feat_dims = args.scene_feat_dims
config.grasp_dim = args.grasp_dim
config.grasp_samples = args.grasp_samples
if args.augment:
    config.transform = transfom_params

config.normalize = train_dataset.normalize_points
config.crop_radius = args.crop_radius

# Analyze the dataset class stats
num_epochs = args.epochs
# train_class_stats = analyze_dataset_stats(train_dataset)
# valid_class_stats = analyze_dataset_stats(val_dataset)
# config.train_class_stats = train_class_stats
# config.valid_class_stats = valid_class_stats
# print("Number of classes in the train dataset: ", len(train_class_stats))
# print("Number of classes in the valid dataset: ", len(valid_class_stats))

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
model = ApproachNet(global_feat_dim= config.scene_feat_dims, grasp_dim=args.grasp_dim, num_grasp_sample=args.grasp_samples,
                     device=device).to(device)

config.model_name = model.__class__.__name__

# If we have multiple GPUs, parallelize the model
if torch.cuda.device_count() > 1 and args.multi_gpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    config.used_gpu_count = torch.cuda.device_count()
    model.multi_gpu = True
    model = DataParallel(model)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Create data loader
train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=args.num_workers)
val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=args.num_workers)



print(f"Train data size: {len(train_dataset)}")
print(f"Val data size: {len(val_dataset)}")

# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    total_grasp_loss = 0
    total_approach_loss = 0
    train_grasp_success = 0
    for i, data in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch}/{num_epochs}"):
        optimizer.zero_grad()
        # Forward pass
        data = data.to(device)
        grasp_pred, approach_score_pred, grasp_gt, grasp_loss, approach_loss, approach_points = model(data)

        # Compute the loss
        loss = grasp_loss + approach_loss
        loss.backward()
        # # Update the weights
        optimizer.step()
        total_loss += loss.item()
        total_grasp_loss += grasp_loss.item()
        total_approach_loss += approach_loss.item()
        if epoch % 50 == 0:
            # Calculate the grasp success rate
            pred = grasp_pred.cpu().detach().reshape(-1, 4, 4).numpy()
            gt = grasp_gt.cpu().detach().reshape(-1, 4, 4).numpy()
            train_grasp_success += check_batch_grasp_success(pred, gt,  0.03, np.deg2rad(30))

    if epoch % 50 == 0:
        train_success_rate = train_grasp_success / (len(train_dataset) * args.grasp_samples)
        wandb.log({"Train Grasp Success Rate": train_success_rate}, step=epoch)
    average_loss = total_loss / len(train_data_loader)
    average_grasp_loss = total_grasp_loss / len(train_data_loader)
    average_approach_loss = total_approach_loss / len(train_data_loader)

    # wandb.log({"Train Loss": average_loss}, step=epoch)
    wandb.log({"Train Loss": average_loss, "Train Grasp Loss": average_grasp_loss, "Train Approach Loss": average_approach_loss}, step=epoch)
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
        valid_grasp_success = 0
        high_error_models = []
        for i, val_data in tqdm(enumerate(val_data_loader), total=len(val_data_loader), desc=f"Valid"):
            # vertices, grasp_gt, batch_idx, querry_point = prepare_samples(device, samples)
            val_data = val_data.to(device)
            grasp_pred, approach_score_pred, grasp_gt, grasp_loss, approach_loss, approach_points = model(val_data)

            val_loss = grasp_loss + approach_loss

            if epoch % 50 == 0:
                # Calculate the grasp success rate
                pred = grasp_pred.cpu().detach().reshape(-1, 4, 4).numpy()
                gt = grasp_gt.cpu().detach().reshape(-1, 4, 4).numpy()
                valid_grasp_success += check_batch_grasp_success(pred, gt,  0.03, np.deg2rad(30))

            total_val_loss += val_loss.item()
            total_val_grasp_loss += grasp_loss.item()
            total_val_approach_loss += approach_loss.item()

        average_val_loss = total_val_loss / len(val_data_loader)
        average_val_grasp_loss = total_val_grasp_loss / len(val_data_loader)
        average_val_approach_loss = total_val_approach_loss / len(val_data_loader)
        if epoch % 50 == 0:
            grasp_success_rate = valid_grasp_success / (len(val_dataset) * args.grasp_samples)
            wandb.log({"Valid Grasp Success Rate": grasp_success_rate}, step=epoch)

            # Save the model if the validation loss is low
            if grasp_success_rate < 0.001:
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

    wandb.log({"Val Loss": average_val_loss, "Val Grasp Loss": average_val_grasp_loss, "Val Approach Loss": average_val_approach_loss}, step=epoch)
    print(f"Train Grasp Loss: {average_grasp_loss:.4f} - Train Approach Loss: {average_approach_loss:.4f} \nVal Grasp Loss: {average_val_grasp_loss:4f} - Val Approach Loss {average_val_approach_loss:.4f}")

# Finish wandb run
wandb.finish()