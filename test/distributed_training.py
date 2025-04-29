import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
import os
import argparse

# Simple Dataset
class SimpleDataset(Dataset):
    def __init__(self, size=8000):
        self.size = size
        self.data = torch.randn(size, 3, 224, 224)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Simple Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64 * 222 * 222, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, epochs=1):
    setup(rank, world_size)

    # Create dataset and dataloaders
    dataset = SimpleDataset()
    train_sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset, batch_size=4, sampler=train_sampler)

    model = SimpleModel().cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.cuda(rank), target.cuda(rank)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}")

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, help="Rank of this node in distributed setup", required=True)
    parser.add_argument("--world_size", type=int, help="Total number of processes", required=True)
    args = parser.parse_args()

    train(args.rank, args.world_size)
