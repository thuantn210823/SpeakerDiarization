import torch
from torch import nn
from SA_EEND import SA_EEND

import sys

class SpeakerEmbLoss(nn.Module):
    def __init__(self,
                 n_speakers: int,
                 n_all_speakers: int,
                 d_spk: int):
        super().__init__()
        self.n_speakers = n_speakers
        self.n_all_speakers = n_all_speakers
        self.d_spk = d_spk
        self.all_n_speakers_arr = torch.arange(0, self.n_all_speakers)
        self.embed = nn.Embedding(self.n_all_speakers, self.d_spk)
        self.alpha = nn.Parameter(torch.rand(1)[0] + torch.Tensor([0.5])[0])
        self.beta = nn.Parameter(torch.rand(1)[0] + torch.Tensor([0.5])[0])
    
    def calc_spk_loss(self, 
                        spk_vecs: torch.Tensor, 
                        spk_indices: torch.Tensor) -> torch.Tensor:
        """
        spk_vecs: predicted speaker vectors, which has the shape (N, S_local, d_spk)
        spk_indices: ground-truth speaker indices, which has the shape (N, S_local)
            silent speaker has the index -1
        Returns:
            torch.Tensor
        """
        N, S_local, d_spk = spk_vecs.shape
        embeds = self.embed(self.all_n_speakers_arr)                                    # (M, d_spk)
        # Noramlize learnable global speaker embedding
        embeds = embeds/torch.norm(embeds, dim = -1, keepdim = True)                    # (M, d_spk)
        spk_vecs = spk_vecs.reshape(-1, self.d_spk).contiguous()                        # (N*S_local, d_spk)
        dists = torch.cdist(spk_vecs, embeds)                                           # (N*S_local, M)
        d = torch.clamp(dists*self.alpha, min = sys.float_info.epsilon) + self.beta     # (N*S_local, M)
        loss = -nn.functional.log_softmax(-d, dim = -1)                                 # (N*S_local, M)
        loss = loss.reshape(N, S_local, -1).contiguous()                                # (N, S_local, M)
        non_silent_filter = torch.where(spk_indices != -1)                              # Tuple(non_silent_spks,non_silent_spks)
        non_silent_indices = spk_indices.flatten()
        non_silent_indices = non_silent_indices[non_silent_indices != -1]               # non_silent_spks
        loss = torch.sum(loss[non_silent_filter[0], non_silent_filter[1]]\
                         [torch.arange(0, len(non_silent_indices)), non_silent_indices])
        return loss/(N*S_local)

class EEND_VC(SA_EEND):
    def __init__(self, 
                 d_spk: int,
                 n_all_speakers: int,
                 silent_spk_thres: float = 0.05,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.d_spk = d_spk
        self.silent_spk_thres = silent_spk_thres
        self.spk_embedding = nn.Linear(self.d_model, d_spk*self.n_speakers)
        self.spk_loss = SpeakerEmbLoss(n_speakers = self.n_speakers,
                                       n_all_speakers = n_all_speakers,
                                       d_spk = d_spk)
    
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
        di_out = self.classifier(out)
        spk_out = self.spk_embedding(out).reshape(di_out.shape[0], di_out.shape[1], self.n_speakers, self.d_model)
        di_prob = torch.sigmoid(di_out).unsqueeze(-1)
        spk_out = (di_prob*spk_out).sum(dim = 1)
        spk_out = spk_out/torch.norm(spk_out, dim = -1, keepdim = True)
        return di_out, spk_out