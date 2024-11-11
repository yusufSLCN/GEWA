import numpy as np
from tpp_dataset import TPPDataset
from create_tpp_dataset import save_contactnet_split_samples
from torch_geometric.loader import DataLoader


if __name__ == "__main__":
    # dataset = ShapeNet(root='data/ShapeNet', categories=['Mug'], split='val', pre_transform=T.KNNGraph(k=6))


    # train_paths, val_paths = save_split_samples('../data', 400, dataset_name="tpp_effdict", contactnet_split=True)
    # train_paths, val_paths = save_split_samples('../data', 400, dataset_name="tpp_effdict_nomean_wnormals", contactnet_split=True)
    train_paths, val_paths = save_contactnet_split_samples('../data', num_mesh=1200, dataset_name="tpp_effdict_nomean_wnormals")
    
    # val_paths = [val_paths[args.sample_idx]]
    dataset = TPPDataset(train_paths, return_pair_dict=True, normalize=True)
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    data_classes = {}
    for data in dataset:
        if data.sample_info['class'] not in data_classes:
            data_classes[data.sample_info['class']] = 1
        else:
            data_classes[data.sample_info['class']] += 1

    print(f"Number of classes: {len(data_classes)}")
    #plot the histogram of the classes
    # import matplotlib.pyplot as plt
    # plt.bar(data_classes.keys(), data_classes.values())
    # plt.show()


