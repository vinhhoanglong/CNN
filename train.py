from model.CNN import CNN
import torch
import torch.nn as nn
import argparse
import datetime
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig,OmegaConf
import os

# Define transforms including data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data Augmentation (Random flip)
    transforms.RandomCrop(32, padding=4),  # Data Augmentation (Random crop)
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

@hydra.main(config_path='config', config_name='config')
def train(cfg: DictConfig):
    # Check if CUDA is available and set the device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(OmegaConf.to_yaml(cfg))
    model = CNN(cfg.dropout).to(device)  
    criterion = hydra.utils.instantiate(cfg.criterion)
    optimizer = hydra.utils.instantiate(cfg.optimizers, params=model.parameters())

    for epoch in range(cfg.num_epoch):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)  # Move data to the device

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{cfg.num_epoch}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
    
    save_dir = 'saved'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the model with a timestamped filename
    torch.save(model.state_dict(), f'{save_dir}/model.save')


if __name__ == "__main__":
    train()
