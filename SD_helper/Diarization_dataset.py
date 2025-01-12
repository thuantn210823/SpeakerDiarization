#!/usr/bin/env python3

# Copyright 2019 Hitachi, Ltd. (author: Yusuke Fujita)
# Modified by: Yexin Yang
# Licensed under the MIT license.
#
from typing import Optional, Callable
import torch
import numpy as np
import librosa

from tqdm import tqdm
import os

def stft(data: np.array,
         win_length: int = 1024,
         hop_length: int = 256):
    """
    Compute STFT features
    data: np.array, shape (n_samples, )
    Returns:
    """
    # round up to nearest power of 2
    fft_size = 1 << (win_length - 1).bit_length()
    if len(data) % win_length == 0:
        return librosa.stft(data, n_fft = fft_size, win_length = win_length, hop_length = hop_length).T[:-1]
    else:
        return librosa.stft(data, n_fft = fft_size, win_length = win_length, hop_length = hop_length).T

def transform(
        Y,
        transform_type=None,
        dtype=np.float32):
    """ Transform STFT feature
    Args:
        Y: STFT
            (n_frames, n_bins)-shaped np.complex array
        transform_type:
            None, "log"
        dtype: output data type
            np.float32 is expected
    Returns:
        Y (numpy.array): transformed feature
    """
    Y = np.abs(Y)
    if not transform_type:
        pass
    elif transform_type == 'log':
        Y = np.log(np.maximum(Y, 1e-10))
    elif transform_type == 'logmel':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 16000
        n_mels = 40
        mel_basis = librosa.filters.mel(sr = sr, 
                                        n_fft = n_fft, 
                                        n_mels = n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
    elif transform_type == 'logmel23':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 16000
        n_mels = 23
        mel_basis = librosa.filters.mel(sr = sr, 
                                        n_fft = n_fft, 
                                        n_mels = n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
    elif transform_type == 'logmel23_mn':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 16000
        n_mels = 23
        mel_basis = librosa.filters.mel(sr = sr, 
                                        n_fft = n_fft, 
                                        n_mels = n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        mean = np.mean(Y, axis=0)
        Y = Y - mean
    elif transform_type == 'logmel23_swn':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 16000
        n_mels = 23
        mel_basis = librosa.filters.mel(sr = sr, 
                                        n_fft = n_fft, 
                                        n_mels = n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        powers = np.sum(Y, axis=1)
        th = (np.max(powers) + np.min(powers)) / 2.0
        for i in range(10):
            th = (np.mean(powers[powers >= th]) + np.mean(powers[powers < th])) / 2
        mean = np.mean(Y[powers > th, :], axis=0)
        Y = Y - mean
    elif transform_type == 'logmel23_mvn':
        n_fft = 2 * (Y.shape[1] - 1)
        sr = 16000
        n_mels = 23
        mel_basis = librosa.filters.mel(sr = sr, 
                                        n_fft = n_fft, 
                                        n_mels = n_mels)
        Y = np.dot(Y ** 2, mel_basis.T)
        Y = np.log10(np.maximum(Y, 1e-10))
        mean = np.mean(Y, axis=0)
        Y = Y - mean
        std = np.maximum(np.std(Y, axis=0), 1e-10)
        Y = Y / std
    else:
        raise ValueError('Unknown transform_type: %s' % transform_type)
    return Y.astype(dtype)


def subsample(Y, T, subsampling=1):
    """ Frame subsampling
    """
    Y_ss = Y[::subsampling]
    T_ss = T[::subsampling]
    return Y_ss, T_ss


def splice(Y, context_size=0):

    """ Frame splicing



    Args:

        Y: feature

            (n_frames, n_featdim)-shaped numpy array

        context_size:

            number of frames concatenated on left-side

            if context_size = 5, 11 frames are concatenated.



    Returns:

        Y_spliced: spliced feature

            (n_frames, n_featdim * (2 * context_size + 1))-shaped

    """

    Y_pad = np.pad(
        Y,
        [(context_size, context_size), (0, 0)],
        'constant')
    Y_spliced = np.lib.stride_tricks.as_strided(
        np.ascontiguousarray(Y_pad),
        (Y.shape[0], Y.shape[1] * (2 * context_size + 1)),
        (Y.itemsize * Y.shape[1], Y.itemsize), writeable=False)
    return Y_spliced

def _count_frames(data_len, size, step):
    return int((data_len - size + step)/step)

def _gen_frame_indices(data_length,
                       size = 2000,
                       step = 2000,
                       use_last_samples = False,
                       label_delay = 0,
                       subsampling = 1):
    i = -1
    for i in range(_count_frames(data_length, size, step)):
        yield i*step, i*step + size
    if use_last_samples and i*step + size < data_length:
        if data_length - (i+1)*step - subsampling*label_delay>0:
            yield (i+1)*step, data_length

def get_labeledSTFT(sample,
                    start: int,
                    end: int,
                    win_length: int,
                    hop_length: int,
                    n_speakers: Optional[int] = None):
    data = sample['audio']['array'][start*hop_length: end*hop_length]
    rate = sample['audio']['sampling_rate']
    Y = stft(data, win_length = win_length, hop_length = hop_length)
    speakers = np.unique(sample['speakers']).tolist()
    if n_speakers is None:
        n_speakers = len(speakers)
    else:
        n_speakers = max(n_speakers, len(speakers))
    T = np.zeros((Y.shape[0], n_speakers), dtype = np.int32)

    for st, et, speaker in zip(sample['timestamps_start'],
                               sample['timestamps_end'],
                               sample['speakers']):
        start_frame = np.rint(st*rate/hop_length).astype(np.int32)
        end_frame = np.rint(et*rate/hop_length).astype(np.int32)
        rel_start = rel_end = None
        if start <= start_frame and start_frame < end:
            rel_start = start_frame - start
        if start < end_frame and end_frame <= end:
            rel_end = end_frame - start
        if rel_start is not None or rel_end is not None:
            T[rel_start:rel_end, speakers.index(speaker)] = 1
    speakers = np.array(speakers)
    sumed_T = np.sum(T, axis = 0)
    filter_T = np.where(sumed_T != 0)[0]
    chunked_speakers = speakers[filter_T]
    chunked_T = T[:, filter_T]
    chunked_T = np.pad(chunked_T, ((0, 0), (0, n_speakers - chunked_T.shape[1])))
    return Y, chunked_T, chunked_speakers

class DiLibriSM(torch.utils.data.Dataset):
    def __init__(self,
                 audio_directory: str,
                 annotation_directory: str):
        super().__init__()
        self.audio_dir = audio_directory
        self.ann_dir = annotation_directory
        self.data = os.listdir(self.ann_dir)

    def __getitem__(self, idx):
        sample = {'audio': {},
                  'timestamps_start': [],
                  'timestamps_end': [],
                  'speakers': []}
        with open(os.path.join(self.ann_dir, self.data[idx]), 'r') as f:
            for line in f.readlines():
                seg_type, file_id, channel_id, start, dur, _, _, speaker, _, _ = line.split(' ')
                sample['timestamps_start'].append(float(start))
                sample['timestamps_end'].append(float(dur) + float(start))
                sample['speakers'].append(speaker)
        filepath = os.path.join(self.audio_dir, file_id + '.wav')
        wav, sr = librosa.load(filepath, sr = None)
        sample['audio']['array'] = wav
        sample['audio']['sampling_rate'] = sr
        return sample
    
    def __len__(self):
        return len(self.data)

class AMI(torch.utils.data.Dataset):
    def __init__(self,
                 subset: str,
                 media_stream: str,
                 audio_directory: Optional[str] = None,
                 annotation_directory: Optional[str] = None,
                 default_directory: str = './'):
        super().__init__()
        self.subset = subset
        self.media_stream = media_stream
        self.audio_dir = audio_directory
        self.ann_dir = annotation_directory
        self.default_dir = default_directory
        self.data = os.listdir(self.ann_dir)

    def __getitem__(self, idx):
        sample = {'timestamps_start': [],
                  'timestamps_end': [],
                  'speakers': [],
                  'audio': {}}
        with open(os.path.join(self.ann_dir, self.data[idx]), 'r') as f:
            for line in f.readlines():
                seg_type, file_id, channel_id, start, length, _, _, speaker, _, _ = line.split(' ')
                #if len(sample['audio']) == 0:
                #    sample['audio'].append(file_id)
                sample['timestamps_start'].append(float(start))
                sample['timestamps_end'].append(float(start) + float(length))
                sample['speakers'].append(speaker)
        meeting_folder = os.path.join(self.audio_dir, file_id, 'audio')
        filepath = os.path.join(meeting_folder, file_id + f'.{self.media_stream}.wav')
        wav, sr = librosa.load(filepath, sr = None)
        sample['audio']['array'] = wav
        sample['audio']['sampling_rate'] = sr
        return sample
    
    def __len__(self):
        return len(self.data)

class DiarizationDataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset: Callable,
                 chunk_size: int,
                 context_size: int = 0,
                 win_length: int = 1024,
                 hop_length: int = 256,
                 subsampling: int = 1,
                 rate: int = 16000,
                 all_speakers: Optional[list] = None,
                 input_transform: Optional[Callable] = None,
                 use_last_samples: bool = False,
                 label_delay: int = 0,
                 n_speakers: Optional[int] = None,
                 shuffle: bool = False) -> None:
        super().__init__()
        if all_speakers:
            self.all_speakers = all_speakers
        self.context_size = context_size
        self.win_length = win_length
        self.hop_length = hop_length
        self.subsampling = subsampling
        self.input_transform = input_transform
        self.n_speakers = n_speakers
        self.chunk_size = chunk_size
        self.chunk_indices = []
        self.label_delay = label_delay
        self.shuffle = shuffle

        self.data = dataset
        # make chunk indices
        for idx, sample in tqdm(enumerate(dataset)):
            data_len = int(len(sample['audio']['array'])/hop_length/subsampling)
            for st, ed in _gen_frame_indices(
                data_len, chunk_size, chunk_size, use_last_samples,
                label_delay = self.label_delay,
                subsampling = subsampling):
                self.chunk_indices.append((idx, st*self.subsampling, ed*self.subsampling))
        print(len(self.chunk_indices), " chunks")
            
    def __len__(self):
        return len(self.chunk_indices)
        
    def __getitem__(self, idx):
        sample_idx, st, ed = self.chunk_indices[idx]
        Y, T, spks = get_labeledSTFT(self.data[sample_idx], st, ed,
                               win_length = self.win_length,
                               hop_length = self.hop_length,
                               n_speakers = self.n_speakers)
        Y = transform(Y, self.input_transform)
        Y_spliced = splice(Y, self.context_size)
        Y_ss, T_ss = subsample(Y_spliced, T, self.subsampling)

        # If the sample contains more than "self.n_speakers" speakers,
        # extract top-(self.n_speakers) speakers
        if self.n_speakers and T_ss.shape[1] > self.n_speakers:
            selected_speakers = np.argsort(T_ss.sum(axis = 0)[::-1][:self.n_speakers])
            T_ss = T_ss[:, selected_speakers]
            spks = spks[:, selected_speakers]
        if self.all_speakers:
            spk_indices = []
            for spk in spks:
                spk_indices.append(self.all_speakers.index(spk))
            if len(spk_indices) < self.n_speakers:
                spk_indices += [-1]*(self.n_speakers - len(spk_indices))
            return Y_ss, T_ss, spk_indices
        else:
            return Y_ss, T_ss, spks