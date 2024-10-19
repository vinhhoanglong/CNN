from model.CNN import CNN
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
import os

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data Augmentation (Random flip)
    transforms.RandomCrop(32, padding=4),  # Data Augmentation (Random crop)
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform) 
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)   

def main(test_loader, model_path):
    model = CNN(0.3)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='saved/CNN_0.001_2024-10-19_20-03-28.save')
    args = parser.parse_args()
    main(test_loader, args.model_path)