import torch
from readable_implementations.utils.config import BasicConfig

class Trainer:
    def __init__(self, config: BasicConfig) -> None:
        self.config = config
        self.loss_fn = config.loss_fn
        self.max_epochs = config.max_epochs
        self.optimizer = config.optimizer
        self.model = config.model
        self.train_dataloader = config.train_dataloader
        self.val_dataloader = config.val_dataloader

    def fit(self) -> None:
        self.optimizer.zero_grad()
        self.train()
        self.optimizer.zero_grad()

    def train(self):
        self.model.train()
        for i in range(self.max_epochs):
            for idx, batch in enumerate(self.train_dataloader):
                out = self.model(**batch)

                loss = self.loss_fn(out, batch["tgt"])
                loss.backward()

                # gradient accumulation
                self.optimizer.zero_grad()
                self.optimizer.step()

                if idx % self.validate_every_batch == 0:
                    self.validate()
                    self.model.train()

        return loss

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        for batch in self.val_dataloader:
            with torch.no_grad():  # TODO: Whether it should be here
                out = self.model(
                    src = batch["src"],
                    tgt = batch["tgt"]
                )
                loss = self.loss_fn(out, target)
                val_loss += loss.item()

        val_loss = val_loss / len(self.val_dataloader)
        return {"val_loss": val_loss}
