from typing import Optional

import torch
import torchaudio
import librosa

import os
import collections

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
        sample = collections.defaultdict(list)
        with open(os.path.join(self.ann_dir, self.data[idx]), 'r') as f:
            for line in f.readlines():
                seg_type, file_id, channel_id, start, length, _, _, speaker, _, _ = line.split(' ')
                #if len(sample['audio']) == 0:
                #    sample['audio'].append(file_id)
                sample['timestamps_start'].append(float(start))
                sample['timestamps_end'].append(float(start) + float(length))
                sample['speakers'].append(speaker)
        meeting_folder = os.path.join(self.audio_dir, file_id, 'audio')
        audio = collections.defaultdict(list)
        filepath = os.path.join(meeting_folder, file_id + f'.{self.media_stream}.wav')
        wav, sr = librosa.load(filepath, sr = None)
        audio['array'] = wav
        audio['sampling_rate'] = sr
        sample['audio'] = audio
        return sample
    
    def __len__(self):
        return len(self.data)