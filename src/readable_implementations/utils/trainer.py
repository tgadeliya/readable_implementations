import logging

import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from tqdm.contrib.logging import logging_redirect_tqdm
from tqdm import tqdm

from readable_implementations.utils.config import BasicConfig

logger = logging.getLogger("train")
logging.basicConfig(level=logging.INFO)

def validation_step(model, val_dataloader, loss_fn, step_idx, device):
    val_loss = 0.0

    preds = []
    targets = []
    for batch in tqdm(val_dataloader, desc="validation"):
        input, target = batch
        input = input.to(device)
        target = target.to(device)
        out = model(input)

        loss = loss_fn(out, target).detach().mean().item()
        val_loss += loss

        prob = out.softmax(dim=1)
        pred = torch.argmax(prob, dim=1).tolist()

        preds.extend(pred)
        targets.extend(target.tolist())

    metrics_report = classification_report(
        targets, preds, output_dict=True, zero_division=True
    )
    val_loss = val_loss / len(val_dataloader)
    metrics_dict = {
        "acc": metrics_report["accuracy"],
        "f1-macro": metrics_report["macro avg"]["f1-score"],
        "f1-micro": metrics_report["weighted avg"]["f1-score"],
        "val_loss": val_loss,
    }

    logger.info(
        f"Validation at {step_idx} step: "
        + " , ".join(f"{key}: {value:.2f}" for key, value in metrics_dict.items())
    )
    return {("validation/" + k): v for k, v in metrics_dict.items()}


class Trainer:
    def __init__(self, config: BasicConfig) -> None:
        self.config = config

        self.device = config.device
        self.validate_every_batch = config.validate_every_batch
        self.loss_fn = config.loss_fn
        self.max_epochs = config.max_epochs
        self.optimizer = config.optimizer
        self.model = config.model
        self.train_dataloader = config.train_dataloader
        self.batch_size = self.train_dataloader.batch_size
        self.val_dataloader = config.val_dataloader

        self.writer = SummaryWriter()
        self.global_step = 0
        self.log_every = config.log_every

        self.val_step = validation_step

    def fit(self) -> None:
        self.optimizer.zero_grad()
        self.train()
        self.optimizer.zero_grad()

    def train(self):
        self.model.train()
        for epoch in range(self.max_epochs):
            self.train_one_epoch(epoch)

    def train_one_epoch(
        self, epoch_idx: int
    ):
        loss_per_logging = 0
        for idx, batch in enumerate(tqdm(self.train_dataloader, f" Epoch {epoch_idx}")):
            self.global_step += self.batch_size
            self.optimizer.zero_grad()
            loss_step = self.train_step(batch)
            self.optimizer.step()
            loss_per_logging += loss_step.item()

            if idx % self.validate_every_batch == (self.validate_every_batch - 1):
                self.validate()

            if idx % self.log_every == (self.log_every - 1):
                loss_per_logging_mean = loss_per_logging / (self.log_every * self.batch_size)
                self.writer.add_scalar("train/loss", loss_per_logging_mean, self.global_step)
                with logging_redirect_tqdm():
                    progress = idx / len(self.train_dataloader) * 100
                    logger.info(
                        f"E:{epoch_idx + 1} ({progress:.0f}%), step={idx + 1}, loss={(loss_per_logging_mean):.3f}"
                    )
                    loss_per_logging = 0.0

    def validate(self):
        self.model.eval()
        val_metrics = self.val_step(
            self.model, self.val_dataloader, self.loss_fn, self.global_step, self.device
        )
        self.model.train()
        for metric, val in val_metrics.items():
            self.writer.add_scalar(metric, val, self.global_step)

    def train_step(self, batch):
        input, target = batch
        # self.tensors_to_device(input, target)
        out = self.model(input)
        loss = self.loss_fn(out, target)
        loss.backward()
        return loss

    def tensors_to_device(self, *tensors):
        for a in tensors:
            a.to(self.device)
