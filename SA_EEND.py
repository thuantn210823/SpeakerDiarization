import torch
from torch import nn

from SD_helper.Embedding import PositionalEncoding
from SD_helper.Transformer import CustomTransformerEncoder, CustomTransformerEncoderLayer

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
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.dim_ffn = dim_ffn
        self.num_layers = num_layers
        self.n_speakers = n_speakers
        self.use_pos_enc = use_pos_enc
        
        self.linear = nn.Linear(input_dim, d_model)
        self.layernorm = nn.LayerNorm(d_model)
        encoder_layer = CustomTransformerEncoderLayer(d_model = d_model,
                                                      n_head = nhead,
                                                      dim_feedforward = dim_ffn,
                                                      dropout = dropout,
                                                      *args, **kwargs)
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model, dropout, max_len = 10000)
        self.encoder = CustomTransformerEncoder(encoder_layer,
                                             num_layers = num_layers)
        self.classifier = nn.Linear(d_model, n_speakers)
    
    def forward(self, 
                X:torch.Tensor, 
                *args, **kwargs):
        """
        X: (N, T, D)
        """
        out = self.linear(X)
        out = self.layernorm(out)
        if self.use_pos_enc:
            out = self.pos_enc(out)
        out = self.encoder(out,
                           *args, **kwargs)
        out = self.classifier(out)
        return out