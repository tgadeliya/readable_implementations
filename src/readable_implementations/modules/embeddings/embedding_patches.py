import torch
from torch.nn import Module

from readable_implementations.modules.utils import Linear
from .utils import init_positional_embeddings


class EmbeddingPatches(Module):
    def __init__(self, patch_size, hidden_size, num_patches):
        super().__init__()
        flatten_patch_size = patch_size * patch_size * 3
        self.linear_projection = Linear(flatten_patch_size, hidden_size)
        self.pe = init_positional_embeddings(
            max_model_len=num_patches, d_emb=hidden_size
        ).T

    def forward(self, inp):
        """
        B x N x (C*P*P) => B x N x D
        """
        out = self.linear_projection(inp)
        out = out + self.pe  # broadcast PE to every example in batch
        return out
