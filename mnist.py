import dataclasses
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import torchvision


# class SimpleModel(nn.Module):

#     def __init__(self):
#         self.linear1 = nn.ModuleDict()
#         nn.Linear(28*28, 256)
#         self.
#         self.linear2 = nn.Linear(256, 10)


@dataclasses.dataclass
class DefaultConfig:
    batch_size: int = 2048
    num_epochs: int = 1000
    log_steps: int = 100


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    config = DefaultConfig()
    simple_model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    ).to(device)
    train_dataset = torchvision.datasets.MNIST(
        "data",
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )
    train_dataloader = data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    iters = 0
    for i in range(config.num_epochs):
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            preds = simple_model(X)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iters % config.log_steps == 0:
                print(f"Epoch: {i} \t Loss: {round(loss.item(), 3)}")
            iters += 1


if __name__ == "__main__":
    main()
