from readable_implementations.modules import (
    TransformerEncoder,
    TransformerDecoder,
    Embeddings,
    Linear,
)
from readable_implementations.modules.utils import Softmax
from torch.nn import Module


class Transformer(Module):
    def __init__(
        self,
        d_emb: int = 768,
        d_model: int = 768,
        encoder_vocab_size: int = 30000,
        decoder_vocab_size: int = 30000,
        max_model_len: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_encoder_heads: int = 8,
        num_decoder_heads: int = 8,
        d_ff: int = 2048,
        num_labels: int = 30000,
    ):
        super().__init__()
        self.input_embeddings = Embeddings(d_emb, encoder_vocab_size, max_model_len)
        self.output_embeddings = Embeddings(d_emb, decoder_vocab_size, max_model_len)
        self.encoder = TransformerEncoder(
            num_blocks=num_encoder_layers,
            num_heads=num_encoder_heads,
            d_model=d_model,
            d_ff=d_ff,
        )
        self.decoder = TransformerDecoder(
            num_blocks=num_decoder_layers,
            num_heads=num_decoder_heads,
            d_model=d_model,
            d_ff=d_ff,
        )
        if num_labels is None:
            # default task - machine translation
            num_labels = decoder_vocab_size
        self.linear = Linear(in_dim=d_model, out_dim=num_labels)
        self.softmax = Softmax(dim=1)

    def forward(self, inp, out):
        inp = self.input_embeddings(inp)
        enc_output = self.encoder(inp)
        out = self.output_embeddings(out)
        dec_output = self.decoder(x=out, encoder_output=enc_output)
        out = self.linear(dec_output)
        probs = self.softmax(out)
        return probs

