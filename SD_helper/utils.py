from typing import Optional

import torch
from torch import nn
import numpy as np

import os
import scipy
import collections
from tqdm import tqdm
from itertools import combinations

from .Clustering import ConstrainedAHC
from .cop_kmeans import cop_kmeans
from .Diarization_dataset import stft, transform, splice, get_labeledSTFT, subsample
from .losses import pit_loss, batch_pit_loss, eda_loss

def avg_ckpt(ckpt_directory: str):
    total_ckpt = {'state_dict': collections.defaultdict(list)}
    ckpt_files = os.listdir(ckpt_directory)
    for ckpt_file in ckpt_files:
        ckpt = torch.load(os.path.join(ckpt_directory, ckpt_file), map_location = 'cpu', weights_only = False)
        state_dict = ckpt['state_dict']
        for key, value in state_dict.items():
            total_ckpt['state_dict'][key].append(value)
    for key, value in total_ckpt['state_dict'].items():
        value = torch.stack(value).mean(dim = 0)
        total_ckpt['state_dict'][key] = value
    return total_ckpt

def calc_diarization_error(pred: torch.Tensor,
                           label: torch.Tensor,
                           label_delay: int = 0,
                           length: Optional[int] = None,
                           n_spk: Optional[int] = None,
                           precomputed = False):
    """
    pred: (T, C)
    label: (T, C)
    Returns: dict of diarization error stats
    """
    length = length if length else len(label)
    if n_spk != 0:
        n_spk = n_spk if n_spk else label.shape[1]
    else:
        n_spk = 1
    label = label[:length - label_delay]
    pred = pred[label_delay: length, :n_spk]
    if n_spk > label.shape[1]:
        label = nn.functional.pad(label, (0, n_spk - label.shape[1]), "constant", 0).to(pred)
    elif n_spk < label.shape[1]:
        pred = nn.functional.pad(pred, (0, label.shape[1] - n_spk), "constant", 0).to(pred)
    if not precomputed:
        decisions = torch.sigmoid(pred) > 0.5
    else:
        decisions = pred
    #decisions = pred[label_delay:, ...] > 0.5
    n_ref = label.sum(axis = -1).long()                 # (T,)
    n_sys = decisions.sum(axis = -1).long()             # (T,)
    res = {}
    res['speech_scored'] = (n_ref > 0).sum()
    res['speech_miss'] = ((n_ref > 0)&(n_sys == 0)).sum()
    res['speech_falarm'] = ((n_ref == 0)&(n_sys > 0)).sum()
    res['speaker_scored'] = (n_ref).sum()
    res['speaker_miss'] = torch.max((n_ref - n_sys), torch.zeros_like(n_ref)).sum()
    res['speaker_falarm'] = torch.max((n_sys - n_ref), torch.zeros_like(n_ref)).sum()
    n_map = ((label == 1)&(decisions == 1)).sum(axis = -1)
    res['speaker_error'] = (torch.min(n_ref, n_sys) - n_map).sum()
    res['correct'] = (label == decisions).sum()/label.shape[1]
    res['diarization_error'] = (res['speaker_miss'] + res['speaker_falarm'] + res['speaker_error'])
    res['frames'] = len(label)
    return res

def report_diarization_error(ys: torch.Tensor, 
                             labels: torch.Tensor, 
                             lengths: Optional[torch.Tensor] = None,
                             n_spks: Optional[torch.Tensor] = None):
    """
    Reports diarization errors
    Should be called with torch.no_grad

    Args:
      ys: B-length list of predictions (torch.FloatTensor)
      labels: B-length list of labels (torch.FloatTensor)
    """
    total_der = 0
    if lengths is not None and n_spks is not None:
        for y, t, length, n_spk  in zip(ys, labels, lengths, n_spks):
            stats = calc_diarization_error(y, t, 0, length, n_spk)
            total_der += stats['diarization_error']/stats['speaker_scored'] if stats['speaker_scored'] != 0 else 0
    elif lengths is None and n_spks is not None:
        for y, t, n_spk  in zip(ys, labels, n_spks):
            stats = calc_diarization_error(y, t, 0, None, n_spk)
            total_der += stats['diarization_error']/stats['speaker_scored'] if stats['speaker_scored'] != 0 else 0
    elif lengths is not None and n_spks is None:
        for y, t, length  in zip(ys, labels, lengths):
            stats = calc_diarization_error(y, t, 0, length, None)
            total_der += stats['diarization_error']/stats['speaker_scored'] if stats['speaker_scored'] != 0 else 0
    else:
        for y, t  in zip(ys, labels):
            stats = calc_diarization_error(y, t, 0)
            total_der += stats['diarization_error']/stats['speaker_scored'] if stats['speaker_scored'] != 0 else 0
    return total_der/len(ys)

def di_transform(sample,
                 win_length: int,
                 hop_length: int,
                 input_transform: str,
                 context_size: int,
                 subsampling: int,
                 n_speakers: int):
    Y, T, speakers = get_labeledSTFT(sample,
                                     win_length = win_length,
                                     hop_length = hop_length,
                                     n_speakers = n_speakers)
    Y = transform(Y, input_transform)
    Y_spliced = splice(Y, context_size)
    Y_ss, T_ss = subsample(Y_spliced, T, subsampling)
    if n_speakers and T_ss.shape[1] > n_speakers:
        selected_speakers = np.argsort(T_ss.sum(axis = 0)[::-1][:n_speakers])
        T_ss = T_ss[:, selected_speakers]
        speakers = [selected_speakers]
    return Y_ss, T_ss, speakers

def extract_feature(Y: np.array, 
                    win_length: int,
                    hop_length: int,
                    transform_type: str,
                    context_size: int,
                    subsampling: int):
    Y = stft(Y, win_length = win_length, hop_length = hop_length) 
    Y = transform(Y, transform_type = transform_type)
    Y = splice(Y, context_size = context_size)
    Y = Y[::subsampling]
    return Y

def get_groudtruth(sample,
                   rate: int,
                   hop_length: int,
                   subsampling):
    speakers = np.unique(sample['speakers']).tolist()
    T = np.zeros((len(sample['audio']['array'])//hop_length+1, len(speakers)))
    start = 0
    end = len(T)
    for st, ed, speaker in zip(sample['timestamps_start'],
                               sample['timestamps_end'],
                               sample['speakers']):
        start_frame = np.rint(st*rate/hop_length).astype(np.int32)
        end_frame = np.rint(ed*rate/hop_length).astype(np.int32)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            T[rel_start:rel_end, speakers.index(speaker)] = 1
    T = T[::subsampling]
    return T

### For EEND-VC
def chunking_evaluation(test_dataset,
                        model,
                        chunk_size: int,
                        rate: int,
                        win_length: int,
                        hop_length: int,
                        transform_type: str,
                        context_size: int,
                        subsampling:int,
                        n_speakers: Optional[int] = None,
                        distance: Optional[int] = None,
                        oracle: bool = False,
                        threshold: float = 0.5,
                        median: Optional[int] = None,
                        clustering_method: str = 'Kmeans',
                        export_to_rttms: bool = False):
    total_loss = 0
    total_der = 0
    total_mi = 0
    total_fa = 0
    total_cf = 0
    for idx, sample in tqdm(enumerate(test_dataset)):
        ### Making inference
        speakers = np.unique(sample['speakers']).tolist()
        if n_speakers and len(speakers) > n_speakers:
            continue
        rttm = ''
        temp_spk_ids = []
        cl = []
        spk_embs = []
        array = sample['audio']['array']
        for i in range(len(array)//(chunk_size*rate)+1):
            # Chunking
            chunk_i = array[i*(chunk_size*rate):min(len(array), chunk_size*rate*(i+1))]
            # Feature extraction
            Y_i = extract_feature(chunk_i, win_length, hop_length, transform_type, context_size, subsampling)
            if len(Y_i) < 512:
                Y_i = np.pad(Y_i, ((0, 512 - len(Y_i)), (0, 0)), 'constant', constant_values = -1)
            # Making Inference
            model.eval()
            T_pred, spk_vecs = model(torch.from_numpy(Y_i).unsqueeze(0))
            T_pred = torch.sigmoid(T_pred.squeeze(0).detach())
            spk_vecs = spk_vecs.squeeze(0).detach()
            # Filtering out silent speakers
            #mean_T_pred = T_pred.mean(dim = 0)
            #filter = torch.where(mean_T_pred > 0.15)[0]
            #>#sum_T_pred = (T_pred>threshold).sum(dim = 0)
            #>#filter = torch.where(sum_T_pred > 0)[0]
            #>#T_pred = T_pred[:, filter].numpy()
            #>#spk_vecs = spk_vecs[filter].numpy()
            # Export rttm
            a = np.where(T_pred.numpy() > threshold, 1, 0)
            if median:
                a = scipy.signal.medfilt(a, (median, 1))
            sum_T_pred = (a).sum(axis = 0)
            filter = np.where(sum_T_pred > 0)[0]
            a = a[:, filter]
            spk_vecs = spk_vecs.numpy()[filter]
            chunk_spks = []
            chunk_spk_ids = []
            for spkid, frames in enumerate(a.T):
                frames = np.pad(frames, (1, 1), 'constant')
                changes, = np.where(np.diff(frames, axis = 0) != 0)
                fmt = "SPEAKER {:s} 1 {:7.2f} {:7.2f} <NA> <NA> {:s} <NA>"
                for s, e in zip(changes[::2], changes[1::2]):
                    if clustering_method:
                        spk_id = f'spk{spkid}_chunk{i}'
                    else:
                        spk_id = spkid
                    rttm += fmt.format(f'session_{idx}', 
                                       s*hop_length*subsampling/rate + chunk_size*i,
                                       (e-s)*hop_length*subsampling/rate,
                                       str(spk_id)) + '\n'
                if len(changes[::2]) > 0:
                    chunk_spk_ids.append(spk_id)
                    chunk_spks.append(len(temp_spk_ids) + spkid)
            cl += list(combinations(chunk_spks, r = 2))
            temp_spk_ids += chunk_spk_ids
            spk_embs.append(spk_vecs)
        spk_embs = np.concatenate(spk_embs, axis = 0)
        # Clustering
        if clustering_method == 'Kmeans':
            pred_spk_ids, _ = cop_kmeans(spk_embs, k = n_speakers, ml = [], cl = cl, max_iter = 1000)
        elif clustering_method == 'AHC':
            if oracle:
                centroids = ConstrainedAHC(n_clusters = len(speakers),
                                           ml = [],
                                           cl = cl,
                                           distance_threshold = None).fit(spk_embs)
            else:
                centroids = ConstrainedAHC(n_clusters = n_speakers,
                                           ml = [],
                                           cl = cl,
                                           distance_threshold = distance).fit(spk_embs)
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
        if export_to_rttms:
            with open(f'session_{idx}.rttm', 'w') as f:
                f.write(rttm)

        ### Evaluation
        T = get_groudtruth(sample, rate, hop_length, subsampling)
        # Retrieving the prediction
        rttm = rttm.split('\n')
        speakers = np.unique(pred_spk_ids).tolist()
        T_pred = np.zeros((T.shape[0]*subsampling, len(speakers)))
        start = 0
        end = len(T_pred)
        for line in rttm:
            _, _, _, st, dur, _, _, speaker, _ = line.split()
            st = float(st)
            et = float(st) + float(dur)
            start_frame = np.rint(st*rate/hop_length).astype(np.int32)
            end_frame = np.rint(et*rate/hop_length).astype(np.int32)
            rel_start = rel_end = None
            if start <= start_frame and start_frame < end:
                rel_start = start_frame - start
            if start < end_frame and end_frame <= end:
                rel_end = end_frame - start
            if rel_start is not None or rel_end is not None:
                T_pred[rel_start:rel_end, speakers.index(int(speaker))] = 1
        T_pred = T_pred[::subsampling]
        if T.shape[1] < T_pred.shape[1]:
            T = np.pad(T, ((0, 0), (0, T_pred.shape[1] - T.shape[1])), "constant", constant_values = 0)
        # Evaluation
        torch.manual_seed(42)
        loss, labels, perms = batch_pit_loss(torch.from_numpy(T_pred).unsqueeze(0), torch.from_numpy(T).unsqueeze(0), precomputed = True)
        T = labels[0]
        stats = calc_diarization_error(torch.from_numpy(T_pred), T, precomputed = True)
        total_der += stats['diarization_error']/stats['speaker_scored']
        total_mi += stats['speaker_miss']/stats['speaker_scored']
        total_fa += stats['speaker_falarm']/stats['speaker_scored']
        total_cf += stats['speaker_error']/stats['speaker_scored']
        total_loss += loss
        # print(loss, stats['diarization_error']/stats['speaker_scored'])
    return total_loss/len(test_dataset), total_der/len(test_dataset), total_mi/len(test_dataset), total_fa/len(test_dataset), total_cf/len(test_dataset)

def evaluate_EEND_VC(test_dataset: torch.utils.data.Dataset,
                     model: nn.Module,
                     n_speakers: int,
                     max_length = None,
                     pad = False,
                     *args, **kwargs):
    total_loss = 0
    total_der = 0
    total_mi = 0
    total_fa = 0
    total_cf = 0
    for sample in tqdm(test_dataset):
        Y, T, speakers = di_transform(sample,
                                      n_speakers = n_speakers,
                                      *args, **kwargs)
        if len(speakers) > n_speakers:
            continue
        length = len(Y)
        if max_length:
            Y = Y[:max_length]
            T = T[:max_length]
        Y = torch.tensor(Y)
        if pad and length < max_length:
            Y = nn.functional.pad(Y.T, (0, max_length - length), value = 0).T[:length]
        model.eval()
        torch.manual_seed(42)
        model.eval()
        T_pred, spk_vecs = model(Y.unsqueeze(0))
        T_pred = T_pred.squeeze(0).detach()
        loss, labels, perms = batch_pit_loss(T_pred.unsqueeze(0), torch.Tensor(T).unsqueeze(0))
        T = labels[0]
        stats = calc_diarization_error(T_pred, torch.tensor(T))
        
        total_der += stats['diarization_error']/stats['speaker_scored']
        total_mi += stats['speaker_miss']/stats['speaker_scored']
        total_fa += stats['speaker_falarm']/stats['speaker_scored']
        total_cf += stats['speaker_error']/stats['speaker_scored']
        total_loss += loss
    return total_loss/len(test_dataset), total_der/len(test_dataset), total_mi/len(test_dataset), total_fa/len(test_dataset), total_cf/len(test_dataset)

def evaluate_SA_EEND(test_dataset: torch.utils.data.Dataset,
                     model: nn.Module,
                     n_speakers: int,
                     max_length = None,
                     *args, **kwargs):
    total_loss = 0
    total_der = 0
    total_mi = 0
    total_fa = 0
    total_cf = 0
    for sample in tqdm(test_dataset):
        Y, T, speakers = di_transform(sample,
                                      n_speakers = n_speakers,
                                      *args, **kwargs)
        if len(speakers) > n_speakers:
            continue
        length = len(Y)
        if max_length:
            Y = Y[:max_length]
            T = T[:max_length]
        Y = torch.tensor(Y)
        model.eval()
        T_pred = model(Y.unsqueeze(0)).squeeze(0).detach()
        loss, labels = batch_pit_loss(T_pred.unsqueeze(0), torch.Tensor(T).unsqueeze(0))
        T = labels[0]
        stats = calc_diarization_error(T_pred, torch.tensor(T))

        total_der += stats['diarization_error']/stats['speaker_scored']
        total_mi += stats['speaker_miss']/stats['speaker_scored']
        total_fa += stats['speaker_falarm']/stats['speaker_scored']
        total_cf += stats['speaker_error']/stats['speaker_scored']
        total_loss += loss
    return total_loss/len(test_dataset), total_der/len(test_dataset), total_mi/len(test_dataset), total_fa/len(test_dataset), total_cf/len(test_dataset)

def evaluate_EEND_EDA(test_dataset: torch.utils.data.Dataset,
                      model: nn.Module,
                      n_speakers: int,
                      max_length = None,
                      pad = False,
                      oracle: bool = True,
                      time_shuffle: bool = True,
                      *args, **kwargs):
    total_loss = []
    total_der = []
    total_mi = []
    total_fa = []
    total_cf = []
    correct = 0
    torch.manual_seed(42)
    for sample in tqdm(test_dataset):
        Y, T, speakers = di_transform(sample,
                                      n_speakers = n_speakers,
                                      *args, **kwargs)
        length = len(Y)
        if max_length:
            Y = Y[:max_length]
            T = T[:max_length]
        Y = torch.tensor(Y)
        if pad and length < max_length:
            Y = nn.functional.pad(Y.T, (0, max_length - length), value = 0).T[:length]
        model.eval()
        torch.manual_seed(42)
        T_pred, attr_logits = model(Y.unsqueeze(0), time_shuffle = time_shuffle)
        estimated_nspk = model.estimate(attr_logits.detach())
        T_pred = T_pred.squeeze(0).detach()
        loss, labels = eda_loss(T_pred.unsqueeze(0), attr_logits.detach(), torch.Tensor(T).unsqueeze(0), torch.tensor([len(speakers)]))
        T = labels[0]
        if oracle:
            stats = calc_diarization_error(T_pred, torch.tensor(T), n_spk = len(speakers))
        else:
            stats = calc_diarization_error(T_pred, torch.tensor(T), n_spk = estimated_nspk.item())
        if stats['speaker_scored'] != 0:
            total_der.append(stats['diarization_error']/stats['speaker_scored'])
            total_mi.append(stats['speaker_miss']/stats['speaker_scored'])
            total_fa.append(stats['speaker_falarm']/stats['speaker_scored'])
            total_cf.append(stats['speaker_error']/stats['speaker_scored'])
            total_loss.append(loss)
        if estimated_nspk.item() == len(speakers):
            correct += 1
        #print(length, loss, stats['diarization_error']/stats['speaker_scored'], estimated_nspk, len(speakers))
    return sum(total_loss)/len(total_loss), sum(total_der)/len(total_der), sum(total_mi)/len(total_mi), sum(total_fa)/len(total_fa), sum(total_cf)/len(total_cf), correct/len(test_dataset)