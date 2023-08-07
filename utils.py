import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def separate_data(dataset, target=4): 
    # Create a list of classes you want to combine
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    classes.remove(target)
    target = target

    # Create a mask for each class and store them in a list
    masks = [dataset.targets == i for i in classes]
    mask_target = dataset.targets == target

    # Combine the masks using logical OR operator
    combined_mask = torch.logical_or.reduce(masks)

    # Get indices of elements in the combined dataset
    indices = torch.where(combined_mask)[0]

    # Shuffle indices
    shuffle_indices = torch.randperm(indices.shape[0])

    # Use the shuffled indices to create a new dataset
    combined_dataset = torch.utils.data.Subset(dataset, indices[shuffle_indices])
    target_dataset = torch.utils.data.Subset(dataset, torch.where(mask_target)[0])


    # Create a DataLoader for the combined dataset
    combined_dataloader = DataLoader(combined_dataset, batch_size=128, shuffle=True)
    dataloader_target = DataLoader(target_dataset, batch_size=128, shuffle=True)
        
    return combined_dataloader, dataloader_target

def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if args.dataset == "MNIST":
        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms)
    elif args.dataset =="CIFAR10":
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
        
    dataloader, dataloader_target = separate_data(trainset)

    return dataloader, dataloader_target


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)