import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_len: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.P = torch.zeros((max_len, embed_dim))
        X = torch.arange(max_len, dtype = torch.float32).reshape(-1, 1)/\
            torch.pow(10000, torch.arange(0, embed_dim, 2, dtype = torch.float32)/embed_dim)
        self.P[:, 0::2] = torch.sin(X)
        self.P[:, 1::2] = torch.cos(X)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, T, H)
        """
        return self.P[: x.shape[1], :].to(x)

class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)