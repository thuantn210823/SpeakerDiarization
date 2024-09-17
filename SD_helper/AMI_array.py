import torch
import librosa
import numpy as np
import pandas as pd
import os

class AMI_array(torch.utils.data.Dataset):
    def __init__(self,
                 audio_directory: str,
                 annotation_directory: str,
                 meetings: list,
                 subset: str,
                 array: int) -> None:
        """
        Array's index name (i.e. 1, 2)
        """
        super().__init__()
        self.audio_dir = audio_directory
        self.ann_dir = annotation_directory
        self.subset = subset
        self.meetings = meetings
        self.array = array
    
    def __len__(self):
        return len(self.meetings)
    
    def __getitem__(self, idx):
        meeting = self.meetings[idx]
        audio_folder_path = os.path.join(self.audio_dir, meeting, 'audio')
        arrays = []
        for i in range(1, 9):
            wav, sr = librosa.load(os.path.join(audio_folder_path, f'{meeting}.Array{self.array}-0{i}.wav'), sr = None)
            arrays.append(wav)
        X = np.stack(arrays)
        return X, meeting
    
class AMI_array_memmap(torch.utils.data.Dataset):
    def __init__(self,
                 memmap_directory: str,
                 annotation_directory: str,
                 csv_file: list,
                 subset: str,
                 array: int) -> None:
        """
        Array's index name (i.e. 1, 2)
        """
        super().__init__()
        self.memmap_dir = memmap_directory
        self.ann_dir = annotation_directory
        self.subset = subset
        self.array = array
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        meeting = row['meeting']
        length = row['length']
        # Get audio data
        audio = {'sampling_rate': 16000}
        audio_folder_path = os.path.join(self.memmap_dir, meeting, 'audio')
        X = np.memmap(os.path.join(audio_folder_path, f'{meeting}.Array{self.array}.dat'), 
                      dtype = 'float32',
                      mode = 'r',
                      shape = (8, int(length)))
        audio['array'] = X
        
        # Get annotation
        sample = {'audio': audio,
                  'timestamps_start': [],
                  'timestamps_end': [],
                  'speakers': []}
        with open(os.path.join(self.ann_dir, f'{meeting}.rttm'), 'r') as f:
            for line in f.readlines():
                seg_type, file_id, channel_id, start, length, _, _, speaker, _, _ = line.split(' ')
                sample['timestamps_start'].append(float(start))
                sample['timestamps_end'].append(float(start) + float(length))
                sample['speakers'].append(speaker)
        return sample
    