##########################################################3
#   Update: default_attractor 1 -> 0
#   0 speaker ~ 1 speaker when calculating diarization loss
#
from typing import List, Optional

import torch
from torch import nn

import numpy as np

from SD_helper.Embedding import PositionalEncoding
from SD_helper.Transformer import CustomTransformerEncoder, CustomTransformerEncoderLayer

class SA_EEND(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int,
                 nhead: int,
                 dim_ffn: int,
                 num_layers: int,
                 use_pos_enc: bool = False,
                 dropout: float = 0.1,
                 *args, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.dim_ffn = dim_ffn
        self.num_layers = num_layers
        self.use_pos_enc = use_pos_enc

        self.linear = nn.Linear(input_dim, d_model)
        encoder_layer = CustomTransformerEncoderLayer(d_model = d_model,
                                                      n_head = nhead,
                                                      dim_feedforward = dim_ffn,
                                                      dropout = dropout,
                                                      *args, **kwargs)
        if use_pos_enc:
            self.pos_enc = PositionalEncoding(d_model, dropout, max_len = 10000)
        self.encoder = CustomTransformerEncoder(encoder_layer,
                                             num_layers = num_layers)
    
    def forward(self, 
                X:torch.Tensor, 
                *args, **kwargs):
        """
        X: (N, T, D)
        """
        out = self.linear(X)
        if self.use_pos_enc:
            out = self.pos_enc(out)
        out = self.encoder(out, *args, **kwargs)
        #out = self.classifier(out)
        return out
    
class EncoderDecoderAttactor(nn.Module):
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 encoder_dropout: float = 0.0,
                 decoder_dropout: float = 0.0):
        super().__init__()
        self.lstm_encoder = nn.LSTM(input_size = input_size,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    batch_first = True,
                                    dropout = encoder_dropout)
        self.lstm_decoder = nn.LSTM(input_size = input_size,
                                    hidden_size = hidden_size,
                                    num_layers = num_layers,
                                    batch_first = True,
                                    dropout = decoder_dropout)
    
    def forward(self, 
                enc_out: torch.Tensor,
                zeros: torch.Tensor) -> torch.Tensor:
        """
        enc_out: (N, T, D)
        """
        _, (H, C) = self.lstm_encoder(enc_out)
        dec_out, _ = self.lstm_decoder(zeros, (H, C))
        return dec_out

class EEND_EDA(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 d_model: int,
                 nhead: int,
                 dim_ffn: int,
                 num_layers: int,
                 use_pos_enc: bool = False,
                 dropout: float = 0.1,
                 num_lstm_layers: int = 1,
                 encoder_dropout: float = 0.0,
                 decoder_dropout: float = 0.0,
                 max_n_speakers: int = 6,
                 *args, **kwargs):
        super().__init__()
        self.max_n_speakers = max_n_speakers
        self.sa = SA_EEND(input_dim = input_dim,
                          d_model = d_model,
                          nhead = nhead, 
                          dim_ffn = dim_ffn,
                          num_layers = num_layers,
                          use_pos_enc = use_pos_enc,
                          dropout = dropout,
                          *args, **kwargs)
        self.eda = EncoderDecoderAttactor(input_size = d_model,
                                          hidden_size = d_model,
                                          num_layers = num_lstm_layers,
                                          encoder_dropout = encoder_dropout,
                                          decoder_dropout = decoder_dropout)
        self.linear = nn.Linear(d_model, 1)
        # nn.init.kaiming_uniform_(self.linear.weight)
    
    def forward(self, 
                src: torch.Tensor,
                time_shuffle: bool = True,
                *args, **kwargs):
        """
        src: (N, T, D)
        """
        # Get embeddings
        enc_out = self.sa(src, *args, **kwargs)
        # Get attactors
        zeros = torch.zeros(src.shape[0], self.max_n_speakers + 1, enc_out.shape[-1]).to(src)
        if time_shuffle:
            post_enc = []
            for enc in enc_out:
                order = np.arange(enc.shape[0])
                np.random.shuffle(order)
                post_enc.append(enc[order])
            post_enc = torch.stack(post_enc).to(src)
        else:
            post_enc = enc_out
        attrs = self.eda(post_enc, zeros)
        di_out = enc_out.matmul(attrs[:, :-1].transpose(1, 2))      # N, T, max_n_speakers
        return di_out, self.linear(attrs)

    def estimate(self,
                 attractor_logits: torch.Tensor,
                 threshold: float = 0.5):
        """
        src: (N, T, D)
        """
        n_spks = []
        attr_probs = torch.sigmoid(attractor_logits).squeeze(2)     # N, max_n_speakers, 1
        for p in attr_probs:
            estimated_attractors = torch.where(p<threshold)[0]
            if len(estimated_attractors) == 0:
                n_spks.append(0)        # Default n_spk <-: Newest Update: 1->0
            else:
                n_spks.append(max(1, estimated_attractors[0].item())) 
        return torch.tensor(n_spks)

if __name__ == '__main__':
    model = EEND_EDA(input_dim = 345,
                     d_model = 256,
                     nhead = 4,
                     dim_ffn = 2048,
                     num_layers = 4,
                     dropout = 0.1,
                     encoder_dropout = 0.1,
                     decoder_dropout = 0.1)