import argparse
import dataclasses
import os

import torch
from torch import distributed
import torch.nn as nn
from torch.utils import data
import torchvision


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7777"
    distributed.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


@dataclasses.dataclass
class DefaultConfig:
    batch_size: int = 2048
    num_epochs: int = 1000
    log_steps: int = 10


def main(rank, world_size, is_distributed):
    if is_distributed:
        ddp_setup(rank, world_size)
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
    sampler = data.DistributedSampler(train_dataset) if is_distributed else None
    shuffle = False if is_distributed else True
    train_dataloader = data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
    iters = 0
    for epoch in range(config.num_epochs):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            preds = simple_model(X)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iters % config.log_steps == 0:
                log_line = (
                    f"Epoch: {epoch} \t Train data size: {len(train_dataloader)} \t"
                    f"Batch size: {len(X)} \t Loss: {round(loss.item(), 3)}"
                )
                if is_distributed:
                    log_line = f"GPU ID: {rank} \t " + log_line
                print(log_line)
            iters += 1
    if is_distributed:
        distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distributed", action="store_true", default=False)
    args = parser.parse_args()
    if args.distributed:
        world_size = torch.cuda.device_count()
        torch.multiprocessing.spawn(
            main, args=(world_size, args.distributed), nprocs=world_size
        )
    else:
        main(rank=None, world_size=None, is_distributed=args.distributed)
