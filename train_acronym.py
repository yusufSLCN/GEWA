import torch
import torch.nn as nn
import torch_geometric.transforms as T
import wandb
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from acronym_dataset import AcronymDataset
from GraspNet import GraspNet
from create_dataset_paths import save_split_meshes
from acronym_utils import analyze_dataset_stats
import os
import numpy as np

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
parser.add_argument('-sfd', '--scene_feat_dims', type=int, default=1028)
parser.add_argument('-n', '--notes', type=str, default='')
args = parser.parse_args()

# Save the split samples
save_split_meshes(args.data_dir, num_mesh=args.num_mesh)
# Load the datasets
if args.augment:
    rotation_range = (-np.pi/3, np.pi/3)  # full circle range in radians
    translation_range = (-0.3, 0.3)  # translation values range
    transfom_params = {"rotation_range": rotation_range, "translation_range": translation_range}
else:
    transfom_params = None

print("Transform params: ", transfom_params)

train_dataset = AcronymDataset('sample_dirs/train_success_simplified_acronym_meshes.npy', transform=transfom_params)
val_dataset = AcronymDataset('sample_dirs/valid_success_simplified_acronym_meshes.npy')
                   
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
if args.augment:
    config.transform = transfom_params

# Analyze the dataset class stats
num_epochs = args.epochs
train_class_stats = analyze_dataset_stats(train_dataset)
valid_class_stats = analyze_dataset_stats(val_dataset)
config.train_class_stats = train_class_stats
config.valid_class_stats = valid_class_stats
print("Number of classes in the train dataset: ", len(train_class_stats))
print("Number of classes in the valid dataset: ", len(valid_class_stats))

# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

config.device = args.device
print(device)

# Initialize the model
model = GraspNet(scene_feat_dim= config.scene_feat_dims, predictor_out_size=9).to(device)
# if torch.cuda.device_count() > 1 and args.device == 'cuda':
#     model = nn.DataParallel(model)
#     print(f"Using {torch.cuda.device_count()} GPUs")

# Define the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Define the loss function
criterion = torch.nn.MSELoss()

def collate_fn(batch):
    batch_idx = []
    vertices = []
    grasp_gt = []
    querry_point = []
    for i, (points, gt, info) in enumerate(batch):
        vertex_count = points.shape[0]
        batch_idx.extend([i] * vertex_count)
        vertices.append(points)
        grasp_gt.append(gt)
        querry_point.append(info['query_point'])
            
    batch_idx = torch.tensor(batch_idx, dtype=torch.int64)
    vertices = torch.cat(vertices, dim=0)
    grasp_gt = torch.stack(grasp_gt, dim=0)
    querry_point = torch.stack(querry_point, dim=0)

    return vertices, grasp_gt, batch_idx, querry_point

# Create data loader
train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.num_workers)
val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.num_workers)


def prepare_samples(device, samples):
    vertices, grasp_gt, batch_idx, querry_point = samples
    vertices = vertices.to(device)
    grasp_gt = grasp_gt.to(device)
    batch_idx = batch_idx.to(device)
    querry_point = querry_point.to(device).float()
    return vertices, grasp_gt ,batch_idx, querry_point

print(f"Train data size: {len(train_dataset)}")


# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    
    for i, samples in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch}/{num_epochs}"):
        optimizer.zero_grad()
        # Forward pass
        vertices, grasp_gt, batch_idx, querry_point = prepare_samples(device, samples)
        grasp_pred = model(None, vertices, batch_idx, querry_point)
        # Compute the loss
        loss = criterion(grasp_pred, grasp_gt)
        # # Backward pass
        loss.backward()
        # # Update the weights
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_data_loader)
    wandb.log({"Train Loss": average_loss}, step=epoch)
    print(f"Train Loss: {average_loss}")

    # Validation loop
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for batch in tqdm(enumerate(val_data_loader), total=len(val_data_loader), desc=f"Valid"):
            vertices, grasp_gt, batch_idx, querry_point = prepare_samples(device, samples)
            grasp_pred = model(None, vertices, batch_idx, querry_point)
            # Compute the loss
            loss = criterion(grasp_pred, grasp_gt)
            total_val_loss += loss.item()
        average_val_loss = total_val_loss / len(val_data_loader)

        if epoch % 10 == 0:
            model_name = f"GraspNet_nm_{args.num_mesh}__bs_{args.batch_size}"
            model_folder = f"models/{model_name}"
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

            model_file = f"{model_name}_epoch_{epoch}.pth"
            model_path = os.path.join(model_folder, model_file)
            torch.save(model.state_dict(), model_path)
            artifact = wandb.Artifact(model_file, type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

    wandb.log({"Val Loss": average_val_loss}, step=epoch)
    print(f"Val Loss: {average_val_loss}")

# Finish wandb run
wandb.finish()