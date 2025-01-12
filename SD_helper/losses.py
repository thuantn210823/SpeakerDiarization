from typing import Optional

import torch
from torch import nn

from itertools import permutations

def pit_loss(pred: torch.Tensor,
                label: torch.Tensor,
                length: Optional[int] = None,
                n_spk: Optional[int] = None,
                label_delay: int = 0,
                precomputed: bool = False):
    """
    pred: (T, C)
    label: (T, C)
    label_delay = int
    """
    length = length if length else len(label)
    if n_spk != 0:
        n_spk = n_spk if n_spk else pred.shape[1]
    else:
        n_spk = 1
    perms = [list(p) for p in permutations(range(n_spk))]
    if not precomputed:
        losses = torch.stack(
            [nn.functional.binary_cross_entropy_with_logits(
                pred[label_delay: length, :n_spk],
                label[..., p][:length-label_delay, :n_spk], reduction = 'none').sum(dim = -1).sum() for p in perms])
    else:
        losses = torch.stack(
            [nn.functional.binary_cross_entropy(
                pred[label_delay: length, :n_spk],
                label[..., p][:length-label_delay, :n_spk], reduction = 'none').sum(dim = -1).sum() for p in perms])
    min_loss = losses.min()/((length - label_delay)*n_spk)
    min_index = losses.argmin().detach()
    return min_loss, label[..., perms[min_index]], perms[min_index]

def batch_pit_loss(ys: torch.Tensor,
                      ts: torch.Tensor,
                      lengths: Optional[torch.Tensor] = None,
                      n_spks: Optional[torch.Tensor] = None,
                      label_delay: int = 0,
                      precomputed: bool = False):
    """
    ys: (N, T, C)
    ts: (N, T, C)
    """
    if lengths is not None and n_spks is not None:
        loss_w_labels = [pit_loss(y, t, len, n_spk, label_delay, precomputed)
                         for (y, t, len, n_spk) in zip(ys, ts, lengths, n_spks)]
    elif lengths is None and n_spks is not None:
        loss_w_labels = [pit_loss(y, t, None, n_spk, label_delay, precomputed)
                         for (y, t, n_spk) in zip(ys, ts, n_spks)]
    elif lengths is not None and n_spks is None:
        loss_w_labels = [pit_loss(y, t, len, None, label_delay, precomputed)
                         for (y, t, len) in zip(ys, ts, lengths)]
    else:
        loss_w_labels = [pit_loss(y, t, None, None, label_delay, precomputed)
                         for (y, t) in zip(ys, ts)]
    losses, labels, perms = zip(*loss_w_labels)
    loss = torch.stack(losses).sum()
    loss = loss/len(losses)
    return loss, labels, perms

def attractor_loss(attractor_logits: torch.Tensor,
                   n_spks: torch.Tensor):
    """
    attractor_logits: (N, max_n_speakers, 1)
    """
    max_n_speakers = attractor_logits.shape[1] - 1
    #n_spks = torch.tensor(n_spks, device = attractor_logits.device)
    n_spks = n_spks.to(attractor_logits)
    labels = (torch.arange(max_n_speakers + 1, device = attractor_logits.device, dtype = attractor_logits.dtype).expand(
        attractor_logits.shape[0], max_n_speakers + 1) < n_spks.unsqueeze(1)).to(torch.float32)
    masks = (torch.arange(max_n_speakers + 1, device = attractor_logits.device, dtype = attractor_logits.dtype).expand(
        attractor_logits.shape[0], max_n_speakers + 1) < (n_spks.unsqueeze(1) + 1)).to(torch.float32)
    loss = nn.functional.binary_cross_entropy_with_logits(attractor_logits.squeeze(2), labels, reduction = 'none')*masks
    loss = (loss.sum(dim = 1)/(n_spks+1)).mean()
    return loss

def eda_loss(ys: torch.Tensor,
             attractor_logits: torch.Tensor,
             ts: torch.Tensor,
             n_spks: Optional[torch.Tensor] = None,
             alpha: float = 1,
             *args, **kwargs):
    di_loss, labels = batch_pit_loss(ys, ts, 
                                     n_spks = n_spks,
                                     *args, **kwargs)
    attr_loss = attractor_loss(attractor_logits = attractor_logits,
                               n_spks = n_spks)
    return di_loss + alpha*attr_loss, labels