from typing import Optional

import torch
from torch import nn

import copy

from .Attention import RelPartialLearnableMultiheadAttn

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 rel_attn: bool = False,
                 *args, **kwargs):
        map_attn = {
            True: RelPartialLearnableMultiheadAttn,
            False: lambda *args, **kwargs: nn.MultiheadAttention(batch_first = True,
                                                                 *args,
                                                                 **kwargs)
        }
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward
        
        self.masked_smhat = map_attn[rel_attn](embed_dim = d_model,
                                              num_heads = n_head,
                                              dropout = dropout)
        self.layernorm1 = nn.LayerNorm(d_model, *args, **kwargs)
        self.dropout1 = nn.Dropout(0.5)

        self.mhat = map_attn[rel_attn](embed_dim = d_model,
                                       num_heads = n_head,
                                       dropout = dropout)
        self.layernorm2 = nn.LayerNorm(d_model, *args, **kwargs)
        self.dropout2 = nn.Dropout(0.1)

        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(inplace = True),
                                 nn.Linear(dim_feedforward, d_model))
        self.layernorm3 = nn.LayerNorm(d_model, *args, **kwargs)  
        self.dropout3 = nn.Dropout(0.1)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                *args, **kwargs):
        sattn_out, _ = self.masked_smhat(query = tgt,
                                      key = tgt,
                                      value = tgt,
                                      attn_mask = tgt_mask,
                                      key_padding_mask = tgt_key_padding_mask,
                                      *args, **kwargs)
        sattn_out = self.layernorm1(self.dropout1(sattn_out) + tgt)
        enc_out, _ = self.mhat(query = sattn_out,
                            key = memory,
                            value = memory,
                            key_padding_mask = memory_key_padding_mask,
                            *args, **kwargs)
        enc_out = self.layernorm2(self.dropout2(enc_out) + sattn_out)
        ffn_out = self.ffn(enc_out)
        ffn_out = self.layernorm3(self.dropout3(ffn_out) + enc_out)
        return ffn_out
    
class CustomTransformerDecoder(nn.Module):
    def __init__(self,
                 decoder_layer: "CustomTransformerDecoder",
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
      
        self.transformerdecoder_layers = _get_clones(decoder_layer, num_layers)

    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                *args, **kwargs):
        for layer in self.transformerdecoder_layers:
            tgt = layer(tgt, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask, *args, **kwargs)
        return tgt

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int,
                 dim_feedforward: int,
                 dropout: float = 0.1,
                 rel_attn: bool = False,
                 *args, **kwargs):
        map_attn = {
            True: RelPartialLearnableMultiheadAttn,
            False: lambda *args, **kwargs: nn.MultiheadAttention(batch_first = True,
                                                                 *args,
                                                                 **kwargs)
        }
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dim_feedforward = dim_feedforward

        self.mhat = map_attn[rel_attn](embed_dim = d_model,
                                       num_heads = n_head,
                                       dropout = dropout)
        self.layernorm2 = nn.LayerNorm(d_model, *args, **kwargs)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(inplace = True),
                                 nn.Linear(dim_feedforward, d_model))
        self.layernorm3 = nn.LayerNorm(d_model, *args, **kwargs)  
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                *args, **kwargs):
        enc_out, _ = self.mhat(query = src,
                               key = src,
                               value = src,
                               key_padding_mask = src_key_padding_mask,
                               attn_mask = src_mask,
                               *args, **kwargs)
        enc_out = self.layernorm2(self.dropout2(enc_out) + src)
        ffn_out = self.ffn(enc_out)
        ffn_out = self.layernorm3(self.dropout3(ffn_out) + enc_out)
        return ffn_out
    
class CustomTransformerEncoder(nn.Module):
    def __init__(self,
                 encoder_layer: "CustomTransformerEncoder",
                 num_layers: int) -> None:
        super().__init__()
        self.num_layers = num_layers
      
        self.transformerencoder_layers = _get_clones(encoder_layer, num_layers)

    def forward(self,
                src: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None,
                *args, **kwargs):
        for layer in self.transformerencoder_layers:
            src = layer(src, src_mask, src_key_padding_mask, *args, **kwargs)
        return src