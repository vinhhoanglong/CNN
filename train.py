from model.CNN import CNN
import torch
import torch.nn as nn
import argparse
import datetime
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Data Augmentation (Random flip)
    transforms.RandomCrop(32, padding=4),  # Data Augmentation (Random crop)
    transforms.ToTensor(),  # Convert images to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

# Load training and test sets
train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

def train(model, criterion, optimizer, num_epoch, lr):
    for epoch in range(num_epoch):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item()}')
    torch.save(model.state_dict(), f'saved/CNN_{lr}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.save')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_epoch', type=int, default=5)
    args = parser.parse_args()
    model = CNN(args.dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train(model, criterion, optimizer, args.num_epoch, args.lr)
