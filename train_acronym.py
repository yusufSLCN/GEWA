import torch
import torch_geometric.transforms as T
import wandb
from torch.utils.data import DataLoader
from acronym_dataset import AcronymDataset
from GraspNet import GraspNet

# Load the datasets
train_dataset = AcronymDataset('train_success_simplified_acronym_samples.npy')
val_dataset = AcronymDataset('valid_success_simplified_acronym_samples.npy')
                   
# Initialize wandb
wandb.init(project="GEWA")

# Log hyperparameters
config = wandb.config
config.learning_rate = 0.001
config.batch_size = 1
config.dataset = train_dataset.__class__.__name__

batch_size = 512
learning_rate = 0.001
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Define the modelel
model = GraspNet(enc_out_channels= 1028, predictor_out_size=16).to(device)


# Define the optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Define the loss function
criterion = torch.nn.MSELoss()

def collate_fn(batch):
    batch_idx = torch.tensor([], dtype=torch.int64)
    vertices = torch.tensor([], dtype=torch.float32)
    grasp_gt = []
    for i in range(len(batch)):
        points, gt = batch[i]
        vertex_count = points.shape[0]
        #append vertex_count times the index
        batch_idx = torch.cat((batch_idx, torch.full((vertex_count,), i, dtype=torch.int64)))
        vertices = torch.cat((vertices, points), 0)
        grasp_gt.append(gt)
            
    grasp_gt = torch.stack(grasp_gt, 0)

    return vertices, grasp_gt, batch_idx
# Create data loader
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print(f"Train data size: {len(train_dataset)}")
# Define the number of epochs
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, samples in enumerate(train_data_loader):
        optimizer.zero_grad()
        print(f"{i}/{len(train_data_loader)}")
        # Forward pass
        # batch_idx = torch.tensor([], dtype=torch.int64)
        vertices, grasp_gt, batch_idx = samples
        vertices = vertices.to(device)
        grasp_gt = grasp_gt.to(device)
        batch_idx = batch_idx.to(device)
        grasp_pred = model(None, vertices, batch_idx)
        # Compute the loss
        loss = criterion(grasp_pred, grasp_gt)
        # # Backward pass
        loss.backward()
        # # Update the weights
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_data_loader)
    print(f"Epoch {epoch + 1}, Train Loss: {average_loss}", end=", ")
    wandb.log({"Train Loss": average_loss}, step=epoch)

    # Validation loop
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        for batch in val_data_loader:
            vertices, grasp_gt, batch_idx = samples
            vertices = vertices.to(device)
            grasp_gt = grasp_gt.to(device)
            batch_idx = batch_idx.to(device)
            grasp_pred = model(None, vertices, batch_idx)
            # Compute the loss
            loss = criterion(grasp_pred, grasp_gt)
            total_val_loss += loss.item()
        average_val_loss = total_val_loss / len(val_data_loader)
        wandb.log({"Val Loss": average_val_loss}, step=epoch)
    print(f"Val Loss: {average_val_loss}")

# Finish wandb run
wandb.finish()