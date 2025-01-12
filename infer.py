from typing import Optional

import torch
import numpy as np
import librosa

import os
import scipy
from itertools import combinations
import argparse
import yaml

from SD_helper.Clustering import ConstrainedAHC
from SD_helper.cop_kmeans import cop_kmeans
from SD_helper.utils import extract_feature

from SA_EEND import *
from EEND_EDA import *
from EEND_VC import *


def infer(args):
    if args.model == 'SA_EEND':
        model = SA_EEND(input_dim = args.input_dim,
                        d_model = args.d_model,
                        nhead = args.nhead,
                        dim_ffn = args.dim_ffn,
                        num_layers = args.num_layers,
                        n_speakers = args.n_speakers,
                        rel_attn = False)
        
    elif args.model == 'EEND_EDA':
        model= EEND_EDA(input_dim = args.input_dim,
                        d_model = args.d_model,
                        nhead = args.nhead,
                        dim_ffn = args.dim_ffn,
                        num_layers = args.num_layers,
                        max_n_speakers = args.max_n_speakers,
                        rel_attn = False)
        
    elif args.model == 'EEND_VC':
        model = EEND_VC(input_dim = args.input_dim,
                        d_model = args.d_model,
                        nhead = args.nhead,
                        dim_ffn = args.dim_ffn,
                        num_layers = args.num_layers,
                        dropout = 0.1,
                        n_speakers = args.n_speakers,
                        d_spk = args.d_spk,
                        n_all_speakers = args.n_all_speakers,
                        rel_attn = False)
    else:
        raise "The model doesn't exist!!!"
    ckpt = torch.load(args.ckpt_path, map_location = 'cpu', weights_only = False)
    model.load_state_dict(ckpt['state_dict'])

    fname = os.path.split(args.audio_path)[1].split('.')[0]
    array, sr = librosa.load(args.audio_path, sr = args.rate)
    rttm = ''
    temp_spk_ids = []
    cl = []
    spk_embs = []
    if args.chunk_size:
        num_chunks = len(array)//(args.chunk_size*args.rate)+1 \
                    if len(array) % args.chunk_size != 0 else len(array)//(args.chunk_size)
        chunk_size = args.chunk_size
    else:
        num_chunks = 1
        chunk_size = len(array)
    for i in range(num_chunks):
        # Chunking
        chunk_i = array[i*(chunk_size*args.rate):min(len(array), chunk_size*args.rate*(i+1))]
        # Feature extraction
        Y_i = extract_feature(chunk_i, args.win_length, args.hop_length, args.transform_type, args.context_size, args.subsampling)
        if len(Y_i) < 512:
            Y_i = np.pad(Y_i, ((0, 512 - len(Y_i)), (0, 0)), 'constant', constant_values = -1)
        # Making Inference
        model.eval()
        if args.model == 'SA_EEND':
            T_pred = model(torch.from_numpy(Y_i.copy()).unsqueeze(0))
            spk_vecs = None
        elif args.model == 'EEND_EDA':
            T_pred, attr_logits = model(torch.from_numpy(Y_i.copy()).unsqueeze(0))
            estimated_nspk = model.estimate(attr_logits.detach())
            if args.oracle:
                T_pred = T_pred[:,:,:args.n_speakers_oracle]
            else:
                T_pred = T_pred[:,:,:estimated_nspk.item()]
            spk_vecs = None
        else:
            T_pred, spk_vecs = model(torch.from_numpy(Y_i.copy()).unsqueeze(0))
            spk_vecs = spk_vecs.squeeze(0).detach()
        T_pred = torch.sigmoid(T_pred.squeeze(0).detach())
        # Export rttm
        a = np.where(T_pred.numpy() > args.threshold, 1, 0)
        if args.median:
            a = scipy.signal.medfilt(a, (args.median, 1))
        sum_T_pred = (a).sum(axis = 0)
        filter = np.where(sum_T_pred > 0)[0]
        a = a[:, filter]
        if spk_vecs:
            spk_vecs = spk_vecs.numpy()[filter]
        chunk_spks = []
        chunk_spk_ids = []
        for spkid, frames in enumerate(a.T):
            frames = np.pad(frames, (1, 1), 'constant')
            changes, = np.where(np.diff(frames, axis = 0) != 0)
            fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
            for s, e in zip(changes[::2], changes[1::2]):
                if args.clustering_method:
                    spk_id = f'spk{spkid}_chunk{i}'
                else:
                    spk_id = spkid
                rttm += fmt.format(fname, 
                                   s*args.hop_length*args.subsampling/args.rate + chunk_size*i,
                                   (e-s)*args.hop_length*args.subsampling/args.rate,
                                   str(spk_id)) + '\n'
            if len(changes[::2]) > 0:
                chunk_spk_ids.append(spk_id)
                chunk_spks.append(len(temp_spk_ids) + spkid)
        cl += list(combinations(chunk_spks, r = 2))
        temp_spk_ids += chunk_spk_ids
        if spk_vecs:
            spk_embs.append(spk_vecs)
    if spk_vecs:
        spk_embs = np.concatenate(spk_embs, axis = 0)
    # Clustering
    if args.clustering_method == 'Kmeans':
        pred_spk_ids, _ = cop_kmeans(spk_embs, k = args.n_speakers, ml = [], cl = cl, max_iter = 1000)
    elif args.clustering_method == 'AHC':
        if args.oracle:
            centroids = ConstrainedAHC(n_clusters = args.n_speakers_oracle,
                                       ml = [],
                                       cl = cl,
                                       distance_threshold = None).fit(spk_embs)
        else:
            centroids = ConstrainedAHC(n_clusters = None,
                                       ml = [],
                                       cl = cl,
                                       distance_threshold = args.distance_threshold).fit(spk_embs)
        pred_spk_ids = centroids.labels_
    else:
        pred_spk_ids = temp_spk_ids
    # Reassign spk ids
    lines = rttm.split('\n')[:-1]
    rttm = []
    for line in lines:
        line = line.split()
        # speaker
        try:
            spk = int(line[7])
        except ValueError:
            spk = line[7]
        line[7] = str(pred_spk_ids[temp_spk_ids.index(spk)])
        rttm.append(' '.join(line))
    rttm = '\n'.join(rttm)
    with open(f'{fname}.rttm', 'w') as f:
        f.write(rttm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_yaml", type = str)
    parser.add_argument("--audio_path", type = str)
    args = parser.parse_args()
    args_ = yaml.load(open(args.config_yaml, 'rb'), Loader = yaml.SafeLoader)
    args_ = argparse.Namespace(**args_)
    setattr(args_, "audio_path", args.audio_path)
    infer(args_)


