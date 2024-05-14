import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from model import Autoencoder
import wandb
from torch_geometric.loader import DataLoader

knn_k = 6
# Load the validation dataset
val_dataset = ShapeNet(root='data/ShapeNet', categories=['Mug'], split='val', pre_transform=T.KNNGraph(k=knn_k))
train_dataset = ShapeNet(root='data/ShapeNet', categories=['Mug'], split='train', pre_transform=T.KNNGraph(k=knn_k), transform=T.RandomJitter(0.01))
# dataset = GeometricShapes(root='data/GeometricShapes')
# dataset.transform = T.Compose([T.SamplePoints(num=num_points), T.KNNGraph(k=knn_k)])
                   
# Initialize wandb
wandb.init(project="GEWA")

# Log hyperparameters
config = wandb.config
config.learning_rate = 0.001
config.batch_size = 128
config.knn_k = knn_k
config.dataset = train_dataset.__class__.__name__
config.catagories = train_dataset.categories

# Define the modelel
model = Autoencoder(3, 32)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

# Define the loss function
criterion = torch.nn.MSELoss()

# Create data loader
train_data_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

print(len(train_dataset))
# Define the number of epochs
num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_data_loader:
        optimizer.zero_grad()
        # Forward pass
        reconstructed = model(batch.pos, batch.pos, batch.edge_index)
        # Compute the loss
        loss = criterion(reconstructed, batch.pos)
        # Backward pass
        loss.backward()
        # Update the weights
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
            reconstructed = model(batch.pos, batch.pos, batch.edge_index)
            loss = criterion(reconstructed, batch.pos)
            total_val_loss += loss.item()
        average_val_loss = total_val_loss / len(val_data_loader)
        wandb.log({"Val Loss": average_val_loss}, step=epoch)
    print(f"Val Loss: {average_val_loss}")

# Finish wandb run
wandb.finish()