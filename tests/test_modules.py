
import torch
from torch.nn import LayerNorm as LN
import pytest

# TODO: Fix Pylance problems with import
from readable_implementations.modules.embedding import Embeddings
from readable_implementations.modules.attention import MultiHeadAttention
from readable_implementations.modules.utils import Linear, LayerNorm, init_kaiming, Softmax

# @pytest.mark.skip
def test_embeddings():
    d_emb = 4
    vocab_size = 10
    model_max_len = 20

    emb_layer = Embeddings(d_emb, vocab_size, model_max_len)

    assert emb_layer.emb_matrix.size() == (vocab_size, d_emb)

    sample_idx = [[1,2,3], [3,3,3,3]]
    out = emb_layer(sample_idx)
    assert out.size() == (len(sample_idx), max(map(len, sample_idx)), d_emb)

# @pytest.mark.skip
def test_linear():
    bs = 1
    ml = 2
    d = 3
    out_dim = 10

    lin = Linear(d, out_dim)
    x = torch.Tensor(bs, ml, d)
    f = lin(x)
    assert f.size() == (bs, ml, out_dim)  

# @pytest.mark.skip
def test_kaiming():
    t = torch.rand(size=(100, 100))
    t_mean_before = t.mean() 
    init_kaiming(t)
    t_mean_after = t.mean()
    assert t_mean_before != t_mean_after, "Opearion isn't in-place style"
    # for big number of values should approach to mean=0 std = sqrt(2/mode)


#@pytest.mark.skip
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

    assert torch.allclose(out, out_imp)

#@pytest.mark.skip
def test_encoder_attention():
    bs = 2
    max_len = 3
    d_model = 6
    n_heads = 3
    kv = d_model

    x = torch.rand(size=(bs, max_len, d_model), dtype=torch.float32)
    torch_mha = torch.nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=n_heads,
        dropout=0,
    )
    mha = MultiHeadAttention(
        d_model = d_model,
        n_heads = n_heads,
        d_key = d_model ,
        d_value = d_model,
        max_len = max_len,
        is_masked = True) 
    out = mha(x)
    print("MHA: ", out)
    print("T MHA: ", torch_mha(x,x,x)[0])
    assert out.size() == x.size()
    assert mha.d_k == mha.d_v == mha.d_q

#@pytest.mark.skip
def test_softmax():
    s = Softmax(dim=1)
    st = torch.nn.Softmax(dim=1)

    x = torch.rand(size=(2, 5, 5))

    out = s(x)
    out_t = st(x)
    # TODO: Investigate why visually equal outputs isn't equal
    print(out[0][0][0].dtype)
    print(out_t[0][0][0].dtype)
    assert out_t.size() == out.size() == x.size()


