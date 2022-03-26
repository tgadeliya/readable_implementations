import torch
from torch import nn
from torch.nn import functional as f

from readable_implementations.modules import (
    EmbeddingPatches,
    TransformerEncoder,
    Linear,
    TransformerFeedForward,
)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_labels=10,
        patch_size=16,
        num_patches=4,
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        mlp_head_size=3072,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = 1 + num_patches   # [CLS] + Image patches tokens
        self.embeddings = EmbeddingPatches(
            patch_size=patch_size, num_patches=self.num_patches, hidden_size=hidden_size
        )
        self.encoder = TransformerEncoder(
            num_blocks=num_layers,
            num_heads=num_heads,
        )
        self.mlp_head = TransformerFeedForward(
            in_dim=hidden_size, out_dim=num_labels, hidden=mlp_head_size
        )

    def forward(self, inp):
        """
        inp - Batch of images split into patches
        B x N x (P x P x C) => B * N+1 * D
        """

        # B x N x (P x P x C) => B x H x N+1 => B x H x N+1
        inp = self.to_patches(inp).permute(0, 2, 1)
        inp = self.embeddings(inp)
        inp = self.encoder(inp)
        # TODO: Add different representations aggregation
        inp = inp[:, 0, :]  # choose only [CLS] token
        inp = self.mlp_head(inp)
        return inp


    def to_patches(self, inp):
        """
        Split image into patches and add [CLS] token.
        B x N x (P x P x C) => B x H x N+1
        """
        inp = f.unfold(inp, kernel_size=self.patch_size, stride=self.patch_size)
        bs, hidden_size, num_patches = inp.size()
        # Add [CLS] TOKEN, which will be updated with Positional Encoding
        # Interesting, thar order matters, because we assume [CLS] to be
        # the first one token
        assert num_patches == self.num_patches - 1, "Problems with num_patches in ViT"
        inp = torch.cat([torch.zeros(bs, hidden_size, 1), inp], dim=-1)
        return inp
