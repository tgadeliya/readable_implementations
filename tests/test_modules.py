
import torch
from torch.nn import LayerNorm as LN
import pytest

# TODO: Fix Pylance problems with import
from transformers_implementations.modules.embedding import Embeddings
from transformers_implementations.modules.utils import Linear, LayerNorm

@pytest.mark.skip
def test_embeddings():
    d_emb = 4
    vocab_size = 10
    model_max_len = 20

    emb_layer = Embeddings(d_emb, vocab_size, model_max_len)

    assert emb_layer.emb_matrix.size() == (vocab_size, d_emb)

    sample_idx = [[1,2,3], [3,3,3,3]]
    out = emb_layer(sample_idx)
    assert out.size() == (len(sample_idx), max(map(len, sample_idx)), d_emb)

@pytest.mark.skip
def test_linear():
    bs = 1
    in_dim = 2
    out_dim = 3

    lin = Linear(in_dim, out_dim)
    x = torch.Tensor(bs, in_dim)

    f = lin(x)

    assert f.size() == (bs, out_dim)  


def test_ln():
    bs = 2
    d_model = 3
    max_len = 3

    x = torch.rand(size=(bs, max_len, d_model))
    
    ln = LN(x.size()[1:], elementwise_affine=False)
    out = ln(x)
    print(out)

    ln_imp = LayerNorm()
    out_imp = ln_imp(x)
    print(out_imp) 
    assert 1==12

