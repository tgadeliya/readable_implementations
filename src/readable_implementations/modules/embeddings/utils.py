import torch


def init_positional_embeddings(max_model_len: int, d_emb: int):
    # TODO: Add random learnable positional embeddings
    # TODO: Now we should transpose PE every time
    pos_arg = torch.arange(0, max_model_len)
    dim_arg = 10000 ** (torch.div(torch.arange(0, d_emb), 2.0) / d_emb).reshape(-1, 1)
    # perform broadcasting to obtain
    # matrix (d_emb, max_model_len),
    # where each pe[:,i] is a positional embedding
    # for position i
    pe = pos_arg / dim_arg
    # apply sin, cos
    pe[::2] = torch.sin(pe[::2])
    pe[1::2] = torch.cos(pe[1::2])
    return pe
