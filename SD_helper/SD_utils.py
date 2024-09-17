from typing import Optional

import torch
from torch import nn

from itertools import permutations

def pit_loss(pred: torch.Tensor,
             label: torch.Tensor,
             length: Optional[int] = None,
             label_delay: int = 0):
    """
    pred: (T, C)
    label: (T, C)
    label_delay = int
    """
    label_perms = [label[..., list(p)] for p
                   in permutations(range(label.shape[-1]))]
    length = length if length else len(label)
    losses = torch.stack(
        [nn.functional.binary_cross_entropy_with_logits(
            pred[label_delay: length, ...],
            l[:length-label_delay, ...]) for l in label_perms])
    min_loss = losses.min()*(len(label) - label_delay)
    min_index = losses.argmin().detach()
    return min_loss, label_perms[min_index]

def batch_pit_loss(ys: torch.Tensor,
                   ts: torch.Tensor,
                   lengths: Optional[list] = None,
                   label_delay: int = 0):
    """
    ys: (N, T, C)
    ts: (N, T, C)
    """
    if lengths is not None:
        loss_w_labels = [pit_loss(y, t, len, label_delay)
                         for (y, t, len) in zip(ys, ts, lengths)]
    else:
        loss_w_labels = [pit_loss(y, t, None, label_delay)
                         for (y, t) in zip(ys, ts)]
    losses, labels = zip(*loss_w_labels)
    loss = torch.stack(losses).sum()
    n_frames = sum([t.shape[0] for t in ts])
    loss = loss/n_frames
    return loss, labels