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

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-bs','--batch_size', type=int, default=64)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
parser.add_argument('-e', '--epochs', type=int, default=50)
parser.add_argument('-d', '--device', type=str, default='cuda')
parser.add_argument('-nw', '--num_workers', type=int, default=0)
parser.add_argument('-nm', '--num_mesh', type=int, default=10)
parser.add_argument('-dd', '--data_dir', type=str, default='../data')
args = parser.parse_args()

# Save the split samples
save_split_meshes(args.data_dir, num_mesh=args.num_mesh)
# Load the datasets
train_dataset = AcronymDataset('sample_dirs/train_success_simplified_acronym_meshes.npy')
val_dataset = AcronymDataset('sample_dirs/valid_success_simplified_acronym_meshes.npy')
                   
# Initialize wandb
wandb.init(project="GEWA")

# Log hyperparameters
config = wandb.config
config.learning_rate = args.learning_rate
config.batch_size = args.batch_size
config.epoch = args.epochs
config.num_mesh = args.num_mesh
config.data_dir = args.data_dir
config.num_workers = args.num_workers
config.dataset = train_dataset.__class__.__name__


# device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
if args.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

config.device = args.device
print(device)

# Define the modelel
model = GraspNet(enc_out_channels= 1028, predictor_out_size=16).to(device)
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
    querry_point = querry_point.to(device)
    return vertices, grasp_gt ,batch_idx, querry_point

print(f"Train data size: {len(train_dataset)}")
# Define the number of epochs
num_epochs = config.epoch

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for i, samples in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch+1}/{num_epochs}"):
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
    print(f"Epoch {epoch + 1}, Train Loss: {average_loss}", end=", ")

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
    wandb.log({"Val Loss": average_val_loss}, step=epoch)
    print(f"Val Loss: {average_val_loss}")

# Finish wandb run
wandb.finish()