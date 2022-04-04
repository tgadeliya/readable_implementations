from dataclasses import dataclass

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, Optimizer


@dataclass
class BasicConfig:
    device: str
    log_every: int
    optimizer: Optimizer
    model: nn.Module
    loss_fn: nn.Module
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    max_epochs: int
    validate_every_batch: int
