from typing import Optional, Callable, Tuple, List

import torch
from torch import nn

import math

from ASR_utils import _init_weight
from Embedding import PositionalEmbedding

def masked_softmax(x: torch.Tensor, 
                   key_padding_mask: Optional[torch.Tensor] = None,
                   attn_mask: Optional[torch.Tensor] = None):
    """
    x: (N, m, n)
    """
    if key_padding_mask is not None:
        attn_bias = torch.zeros_like(key_padding_mask).to(x)
        attn_bias.masked_fill_(key_padding_mask == True, -1e9)
        x = x.transpose(0, 1) + attn_bias
        x = x.transpose(0, 1)


    if attn_mask is not None:
        attn_bias = torch.zeros_like(attn_mask).to(x)
        attn_bias.masked_fill_(attn_mask == True, -1e9)
        x += attn_bias

    return nn.functional.softmax(x, dim = -1)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, 
                 dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2))/math.sqrt(d)
        attn_weights = masked_softmax(scores, key_padding_mask, attn_mask)
        return torch.bmm(self.dropout(attn_weights), value), attn_weights

class MultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 init_weight: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn_dim = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.attention = ScaledDotProductAttention(dropout)

        if init_weight:
            _init_weight(self.Wq)
            _init_weight(self.Wk)
            _init_weight(self.Wv)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor,
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True) -> torch.Tensor:
        """
        query, key, value: (N, *, embed_dim)
        key_padding_mask: (N, S)
        """
        if self.embed_dim % self.num_heads !=0:
            raise "embed_dim must be divied by num_heads"
        bs = query.shape[0]
        Q = self.Wq(query).reshape(bs, -1, self.num_heads, self.attn_dim).transpose(1, 2).contiguous().reshape(bs*self.num_heads, -1, self.attn_dim)
        K = self.Wk(key).reshape(bs, -1, self.num_heads, self.attn_dim).transpose(1, 2).contiguous().reshape(bs*self.num_heads, -1, self.attn_dim)
        V = self.Wv(value).reshape(bs, -1, self.num_heads, self.attn_dim).transpose(1, 2).contiguous().reshape(bs*self.num_heads, -1, self.attn_dim)
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat_interleave(self.num_heads, dim = 0)

        H, attn_weights = self.attention(Q, K, V, 
                                         key_padding_mask = key_padding_mask,
                                         attn_mask = attn_mask)
        H = H.reshape(bs, self.num_heads, -1, self.attn_dim).transpose(1, 2).contiguous().reshape(bs, -1, self.embed_dim)
        if need_weights:
            return H, attn_weights
        else:
            return H

class RelMultiheadAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 init_weight: bool = True) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.attn_dim = embed_dim//num_heads
        self.dropout = nn.Dropout(dropout)

        self.W_query = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.W_key = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.W_value = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.W_relpos = nn.Linear(embed_dim, embed_dim, bias = bias)
        self.W_out = nn.Linear(embed_dim, embed_dim, bias = bias)

        self.U_bias = nn.Parameter(torch.Tensor(num_heads, self.attn_dim))
        self.V_bias = nn.Parameter(torch.Tensor(num_heads, self.attn_dim))
        
        if init_weight:
            _init_weight(self.W_query)
            _init_weight(self.W_key)
            _init_weight(self.W_value)
            _init_weight(self.W_relpos)
            _init_weight(self.W_out)
        nn.init.xavier_uniform_(self.U_bias)
        nn.init.xavier_uniform_(self.V_bias)

    def _rel_shift(self, x, zero_triu=False):
        zero_pad = torch.zeros((x.size()[0], x.size()[1], x.size()[2], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size()[0], x.size()[1], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)
        if zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :].to(x)
        return x
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                relpos: torch.Tensor,
                zero_triu: bool = False,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True):
        """
        query, key, value, relpos: (N, *, embed_dim)
        Note: The number of keys, values and relpos must be equal. relpos must be in a reversed order if it is non-learnable.
        """
        bs = query.shape[0]
        Q = self.W_query(query).reshape(bs, -1, self.num_heads, self.attn_dim)                                          # N, m, h, d
        K = self.W_key(key).reshape(bs, -1, self.num_heads, self.attn_dim).transpose(1, 2).contiguous()                 # N, h, n, d
        V = self.W_value(value).reshape(bs, -1, self.num_heads, self.attn_dim).transpose(1, 2).contiguous()\
            .reshape(bs*self.num_heads, -1, self.attn_dim)                                                              # N.h, n, d
        R = self.W_relpos(relpos.to(query)).reshape(bs, -1, self.num_heads, self.attn_dim).transpose(1, 2).contiguous() # N, h, n, d

        AC = (Q + self.U_bias).transpose(1, 2).matmul(K.transpose(2, 3))                                                    # N, h, m, n    
        BD = self._rel_shift((Q + self.V_bias).transpose(1, 2).matmul(R.transpose(2, 3)), zero_triu)                        # N, h, m, n
        scores = (AC + BD)/math.sqrt(self.attn_dim)
        scores = scores.reshape(bs*self.num_heads, scores.shape[2], scores.shape[3])                                    # N.h, m, n
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat_interleave(self.num_heads, dim = 0)
        
        attn_weights = masked_softmax(scores,
                                      key_padding_mask = key_padding_mask,
                                      attn_mask = attn_mask)
        
        H = torch.bmm(self.dropout(attn_weights), V)                                                                  # N.h, m, d
        H = H.reshape(bs, self.num_heads, -1, self.attn_dim).transpose(1, 2).reshape(bs, -1, self.embed_dim)
        H = self.W_out(H)

        if need_weights:
            return H, attn_weights
        else:
            return H

class RelPartialLearnableMultiheadAttn(RelMultiheadAttention):
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 dropout: float = 0.0,
                 bias: bool = True,
                 init_weight: bool = True) -> None:
        super().__init__(embed_dim,
                         num_heads,
                         dropout,
                         bias,
                         init_weight)
        self.posemb = PositionalEmbedding(embed_dim)
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                reversed: bool = True,
                zero_triu: bool = False,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True):
        bs = query.shape[0]
        PosEmbMat = self.posemb(key).unsqueeze(0).repeat_interleave(bs, dim = 0).to(query)
        if reversed:
            PosEmbMat = PosEmbMat[torch.arange(0, PosEmbMat.shape[0]).tolist()[::-1], :]
        return super().forward(query,
                               key,
                               value,
                               PosEmbMat,
                               zero_triu,
                               key_padding_mask,
                               attn_mask,
                               need_weights)