import torch
import torch.nn as nn
import torch_geometric.transforms as T
import wandb
from torch_geometric.loader import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch_geometric.transforms import RandomJitter, Compose
import argparse
from tqdm import tqdm
from tpp_dataset import TPPDataset
from EdgeGraspNet import EdgeGrasp
from create_tpp_dataset import save_contactnet_split_samples
from metrics import check_batch_success_with_whole_gewa_dataset, check_batch_grasp_success_rate_per_point
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
parser.add_argument('-as', '--approach_samples', type=int, default=32)
parser.add_argument('-cs', '--contact_samples', type=int, default=32)
parser.add_argument('-li', '--log_interval', type=int, default=10)
parser.add_argument('-oc', '--only_classifier', action='store_true')
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
train_dataset = TPPDataset(train_dirs, transform=transform, return_normals=True, normalize=True)
val_dataset = TPPDataset(val_dirs, return_normals=True, normalize=True)
                   
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

config.approach_samples = args.approach_samples
config.contact_samples = args.contact_samples
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
# max_grasp_per_edge = 10
topk = 10
num_samples = args.contact_samples * args.approach_samples

model = EdgeGrasp(num_app_samples=args.approach_samples, num_contact_samples=args.contact_samples).to(device)
config.model_name = model.__class__.__name__
config.topk = topk
wandb.watch(model, log="all")

multi_gpu = False
# If we have multiple GPUs, parallelize the model
if torch.cuda.device_count() > 1 and args.multi_gpu:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    config.used_gpu_count = torch.cuda.device_count()
    # model.multi_gpu = True
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


print(f"Train data size: {len(train_dataset)}")
print(f"Val data size: {len(val_dataset)}")
# Training loop
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    train_accuracy = 0
    train_recall = 0
    train_precision = 0
    train_f1 = 0
    train_grasp_success = 0
    for i, data in tqdm(enumerate(train_data_loader), total=len(train_data_loader), desc=f"Epoch {epoch}/{num_epochs}"):
        optimizer.zero_grad()
        if not multi_gpu:
            # data = data.to(device)
            data.pos = data.pos.to(device)
            data.batch = data.batch.to(device)
            data.normals = data.normals.to(device)
        train_loss, train_grasp_pred, train_success_pred, train_success_gt, train_approach_points = model(data)


        # num_pos_pred = train_success_pred > 0.5
        # print(f"Loss takes {t4 - t3}s")
        if multi_gpu:
            train_loss = train_loss.mean()

        train_loss.backward()
        # # Update the weights
        optimizer.step()
        total_loss += train_loss.item()


        if epoch % args.log_interval == 0:
            with torch.no_grad():
                train_success_pred = train_success_pred.flatten()
                train_success_gt = train_success_gt.flatten().int()
                train_accuracy += binary_accuracy(train_success_pred, train_success_gt)
                train_recall += binary_recall(train_success_pred, train_success_gt)
                train_precision += binary_precision(train_success_pred, train_success_gt)
                train_f1 += binary_f1_score(train_success_pred, train_success_gt)

                train_success_pred = train_success_pred.cpu().detach().reshape(-1, num_samples)
                # get top 10 highest grasp predictions indexes 
                grasp_success_scores, grasp_pred_idx = torch.topk(train_success_pred, topk, dim=1)

                train_grasp_pred = train_grasp_pred.cpu().detach().reshape(-1, num_samples, 1, 4, 4).numpy()
                batch_size = train_grasp_pred.shape[0]
                topk_grasp_pred = train_grasp_pred[np.arange(batch_size)[:, None], grasp_pred_idx]
                topk_grasp_pred = topk_grasp_pred.reshape(-1, topk, 1, 4, 4)
                # if multi_gpu:
                #     grasp_gt_paths = [s.sample_info['grasps'] for s in data]
                #     means = [s.sample_info['mean'] for s in data]
                # else:
                grasp_gt_paths = [data[i].sample_info['grasps'] for i in range(len(data))]
                means = [data[i].sample_info['mean'] for i in range(len(data))]

                train_grasp_success += check_batch_success_with_whole_gewa_dataset(topk_grasp_pred, 0.03, np.deg2rad(30),grasp_gt_paths, means)

    average_train_loss = total_loss / len(train_data_loader)

    wandb.log({"Average Train Loss": average_train_loss}, step=epoch)
    print(f"Average Train Loss: {average_train_loss:.4f}")

    scheduler.step()
    if epoch % args.log_interval == 0:
        train_success_rate = train_grasp_success / len(train_data_loader)
        wandb.log({"Train Grasp Success Rate": train_success_rate}, step=epoch)

        train_accuracy = train_accuracy / len(train_data_loader)
        train_f1 = train_f1 / len(train_data_loader)
        train_precision = train_precision / len(train_data_loader)
        train_recall = train_recall / len(train_data_loader)
        wandb.log({"Train Recall": train_recall, "Train Accuracy":train_accuracy,
                    "Train Precision":train_precision, "Train_F1":train_f1}, step=epoch)
    # Validation loop
    average_val_loss = 0
    if epoch % args.log_interval == 0:
        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            valid_accuracy = 0
            val_recall = 0
            val_precision = 0
            val_f1 = 0
            val_grasp_success = 0
            for i, val_data in tqdm(enumerate(val_data_loader), total=len(val_data_loader), desc=f"Valid"):
                if not multi_gpu:
                    val_data.pos = val_data.pos.to(device)
                    val_data.batch = val_data.batch.to(device)
                    val_data.normals = val_data.normals.to(device)
                    
                val_loss, val_grasp_pred, val_success_pred, val_success_gt, val_approach_points = model(val_data)

                if multi_gpu:
                    val_loss = val_loss.mean()

                total_val_loss += val_loss.item()
                
                val_success_pred = val_success_pred.flatten()
                val_success_gt = val_success_gt.flatten().int()
                valid_accuracy += binary_accuracy(val_success_pred, val_success_gt)
                val_recall += binary_recall(val_success_pred, val_success_gt)
                val_precision += binary_precision(val_success_pred, val_success_gt)
                val_f1 += binary_f1_score(val_success_pred, val_success_gt)

                val_success_pred = val_success_pred.cpu().detach().reshape(-1, num_samples)
                # get top 10 highest grasp predictions indexes 
                val_grasp_success_scores, val_grasp_pred_idx = torch.topk(val_success_pred, topk, dim=1)

                val_grasp_pred = val_grasp_pred.cpu().detach().reshape(-1, num_samples, 1, 4, 4).numpy()
                batch_size = val_grasp_pred.shape[0]
                val_topk_grasp_pred = val_grasp_pred[np.arange(batch_size)[:, None], val_grasp_pred_idx]
                val_topk_grasp_pred = val_topk_grasp_pred.reshape(-1, topk, 1, 4, 4)

                # if multi_gpu:
                #     val_grasp_gt_paths = [s.sample_info['grasps'] for s in val_data]
                #     val_means = [s.sample_info['mean'] for s in val_data]
                # else:
                val_grasp_gt_paths = [val_data[i].sample_info['grasps'] for i in range(len(val_data))]
                val_means = [val_data[i].sample_info['mean'] for i in range(len(val_data))]

                val_grasp_success += check_batch_success_with_whole_gewa_dataset(val_topk_grasp_pred, 0.03, np.deg2rad(30), val_grasp_gt_paths, val_means)



            average_val_loss = total_val_loss / len(val_data_loader)
            val_grasp_success_rate = val_grasp_success / len(val_data_loader)

            wandb.log({"Valid Grasp Success Rate": val_grasp_success_rate}, step=epoch)

            valid_accuracy = valid_accuracy / len(val_data_loader)
            val_recall = val_recall / len(val_data_loader)
            val_precision = val_precision / len(val_data_loader)
            val_f1 = val_f1 / len(val_data_loader)
            wandb.log({"Val Recall": val_recall, "Val Accuracy":valid_accuracy, 
                       "Val Precision":val_precision, "Val_F1":val_f1}, step=epoch)
            
            print(f"Train Acc: {train_accuracy} - Valid Accuracy: {valid_accuracy}")
            print(f"Train Recall: {train_recall} - Valid Recall: {val_recall}")
            print(f"Train Precision: {train_precision} - Valid Precision: {val_precision}")
            print(f"Train F1: {train_f1} - Valid F1: {val_f1}")
            print(f"Train Grasp Success: {train_success_rate} - Valid Grasp Success: {val_grasp_success_rate}")
            # Save the model if the validation loss is low
            if val_grasp_success_rate > 0.5:
                model_name = f"{config.model_name}_nm_{args.num_mesh}__bs_{args.batch_size}.pth"
                model_folder = f"models/{model_name}"
                if not os.path.exists(model_folder):
                    os.makedirs(model_folder)

                model_file = f"{model_name}_epoch_{epoch}_success_{val_grasp_success_rate:.2f}_acc_{valid_accuracy:.2f}_recall_{val_recall:.2f}.pth"
                model_path = os.path.join(model_folder, model_file)
                torch.save(model.state_dict(), model_path)
                artifact = wandb.Artifact(model_file, type='model')
                artifact.add_file(model_path)
                wandb.log_artifact(artifact)

        wandb.log({"Val Loss": average_val_loss}, step=epoch)
        print(f"Train Loss: {average_train_loss:.4f} - Val Loss: {average_val_loss:.4f}")

# Finish wandb run
wandb.finish()