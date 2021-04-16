from transformers_implementations.modules.embedding import Embeddings

def test_pos_emb():
    d_emb = 4
    vocab_size = 10
    model_max_len = 20

    emb_layer = Embeddings(d_emb, vocab_size, model_max_len)

    assert emb_layer.emb_matrix.size() == (vocab_size, d_emb)

    sample_idx = [[1,2,3], [3,3,3,3]]
    out = emb_layer(sample_idx)
    assert out.size() == (len(sample_idx), max(map(len, sample_idx)), d_emb)