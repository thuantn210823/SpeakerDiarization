import torch
from torch import nn
import torchaudio

from Embedding import PositionalEncoding

class SA_EEND(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int,
                 nhead: int,
                 dim_ffn: int,
                 num_layers: int,
                 n_speakers: int,
                 use_pos_enc: bool = False,
                 dropout: float = 0.1,
                 *args, **kwargs):
        super().__init__()
        self.use_pos_enc = use_pos_enc
        self.linear = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                   nhead = nhead,
                                                   dim_feedforward = dim_ffn,
                                                   batch_first = True,
                                                   dropout = dropout,
                                                   *args, **kwargs)
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model, dropout, max_len = 10000)
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers = num_layers)
        self.layernorm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_speakers)
    
    def forward(self, 
                X:torch.Tensor, 
                *args, **kwargs):
        """
        X: (N, T, D)
        """
        out = self.linear(X)
        if self.use_pos_enc:
            out = self.pos_enc(out)
        out = self.layernorm(self.encoder(out, *args, **kwargs))
        out = self.classifier(out)
        return out