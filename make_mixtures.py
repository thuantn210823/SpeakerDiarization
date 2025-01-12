from typing import Callable

import torch
import librosa
import soundfile as sf
import numpy as np

import math
import os
import scipy.signal as ss
import random
import argparse

class DiLibriSpeech(torch.utils.data.Dataset):
    def __init__(self, 
                 speaker_root: str, 
                 rir_roots: list[str],
                 musan_root: str,
                 spk_dur_stat: str,
                 same_spk_pause_stat: str,
                 diff_spk_pause_stat: str,
                 diff_spk_overlap_stat: str,
                 sr: int = 16000,
                 Nspeakers: int = 2,
                 Numin: int = 10,
                 Numax: int = 20,
                 maxlen: int = 20*15*16000,
                 snrs: list = [10, 15, 20]):
        super().__init__()
        self.speaker_root = speaker_root
        self.speakers = os.listdir(speaker_root)
        self.rooms = rir_roots
        self.musan_root = musan_root
        with open(os.path.join(musan_root, 'noise/free-sound/ANNOTATIONS'), 'r') as f:
            self.bg_noises = f.read().split('\n')[1:-1]
        self.snrs = snrs
        self.sr = sr
        self.maxlen = maxlen
        self.Numin = Numin
        self.Numax = Numax
        self.Nspeakers = Nspeakers
        self.speaker_dur = np.load(spk_dur_stat)
        self.speaker_dur[:, 1] = self.speaker_dur[:, 1]/np.sum(self.speaker_dur[:, 1])
        self.same_spk_pause = np.load(same_spk_pause_stat)
        self.same_spk_pause[:, 1] = self.same_spk_pause[:, 1]/np.sum(self.same_spk_pause[:, 1])
        self.diff_spk_pause = np.load(diff_spk_pause_stat)
        self.diff_spk_pause[:, 1] = self.diff_spk_pause[:, 1]/np.sum(self.diff_spk_pause[:, 1])
        self.diff_spk_overlap = np.load(diff_spk_overlap_stat)
        self.diff_spk_overlap[:, 1] = self.diff_spk_overlap[:, 1]/np.sum(self.diff_spk_overlap[:, 1])
        spk_pause = np.sum(self.diff_spk_pause[:, 0])
        overlap = np.sum(self.diff_spk_overlap[:, 0])
        self.diff_spk_pause_vs_overlap_prob = spk_pause / (spk_pause + overlap)

    def __len__(self):
        return len(self.speakers)

    def __getitem__(self, idx):
        ### Choose speakers
        main_speaker = self.speakers[idx]
        aux_speakers = [main_speaker]*self.Nspeakers
        while main_speaker in aux_speakers:
            aux_speakers = np.random.choice(self.speakers, self.Nspeakers - 1)
        speakers = [main_speaker] + aux_speakers.tolist()
        utts = [self._utts_list(os.path.join(self.speaker_root, speaker)) for speaker in speakers]
        room = self.rooms[idx%len(self.rooms)]
        rirs = os.listdir(room)
        
        ann = {'st': [],
               'ed': [],
               'speakers': [],
               'audio': {}}
        L = []
        G = {}
        I = {}
        conv = np.zeros(self.maxlen)
        for i, speaker in enumerate(speakers):
            if len(utts[i]) < self.Numin:
                continue
            random.shuffle(utts[i])
            for utt in utts[i][:self.Numax]:
                if not utt.endswith('.flac'):
                    continue
                L.append(utt)
                G[utt] = speaker
                I[utt] = np.random.choice(rirs)

        random.shuffle(L, self._shuffle_weight)
        utt = L[0]
        speaker, segment, _ = utt.split('-')
        speech, _ = librosa.load(os.path.join(self.speaker_root, speaker, segment, utt), sr = None)
        speech_dur = 2*max(0.5, np.random.choice(self.speaker_dur[:, 0], p = self.speaker_dur[:, 1]))
        speech = speech[-int((speech_dur+0.2)*self.sr):-int(0.2*self.sr)]
        rir_signal, _ = librosa.load(os.path.join(room, I[utt]), sr = None)
        speech = ss.fftconvolve(speech, rir_signal)
        startpos = 0
        endpos = len(speech)
        conv[startpos: endpos] = speech
        ann['st'].append(startpos)
        ann['ed'].append(endpos)
        ann['speakers'].append(G[utt])
        pos = endpos
        for t in range(1, len(L)):
            if G[L[t-1]] == G[L[t]]:
                gap = max(0.025, np.random.choice(self.same_spk_pause[:, 0], p = self.same_spk_pause[:, 1]))
                speech_dur = max(0.5, np.random.choice(self.speaker_dur[:, 0], p = self.speaker_dur[:, 1]))
            elif np.random.binomial(1, self.diff_spk_pause_vs_overlap_prob):
                speech_dur = 2*max(0.5, np.random.choice(self.speaker_dur[:, 0], p = self.speaker_dur[:, 1]))
                gap = max(0.025, np.random.choice(self.diff_spk_pause[:, 0], p = self.diff_spk_pause[:, 1]))
            else:
                st = ann['st'][-1] if t == 1 else max(ann['st'][-1], ann['ed'][-2])
                gap = -min((ann['ed'][-1]-st)/16000,
                           2*max(0.15, np.random.choice(self.diff_spk_overlap[:, 0], p = self.diff_spk_overlap[:, 1])))
                speech_dur = max(-gap,
                                 2*max(0.5, np.random.choice(self.speaker_dur[:, 0], p = self.speaker_dur[:, 1])))
            gap = int(gap*self.sr)
            utt = L[t]
            speaker, segment, _ = utt.split('-')
            speech, _ = librosa.load(os.path.join(self.speaker_root, speaker, segment, utt), sr = None)
            speech = speech[-int((speech_dur+0.2)*self.sr):-int(0.2*self.sr)]
            rir_signal, _ = librosa.load(os.path.join(room, I[utt]), sr = None)
            speech = ss.fftconvolve(speech, rir_signal)
            startpos = max(0, pos + gap)
            endpos = min(startpos + len(speech), len(conv))
            if startpos >= endpos:
                break
            conv[startpos: endpos] += speech[:(endpos - startpos)]
            ### Write label
            ann['st'].append(startpos)
            ann['ed'].append(endpos)
            ann['speakers'].append(G[utt])
            # update pos
            pos = endpos
            if pos >= self.maxlen:
                break
        conv = conv[:pos]
        noise_data, _ = librosa.load(os.path.join(self.musan_root, 'noise/free-sound', np.random.choice(self.bg_noises)) + '.wav', sr = self.sr)
        if pos > len(noise_data):
            noise_data = np.pad(noise_data, (0, pos - len(noise_data)), 'wrap')
        else:
            noise_data = noise_data[:pos]
        # noise power is scaled according to selected SNR, then mixed
        signal_power = np.sum(conv**2) / len(conv)
        noise_power = np.sum(noise_data**2) / len(noise_data)
        scale = math.sqrt(
                    math.pow(10, - np.random.choice(self.snrs) / 10) * signal_power / noise_power)
        conv += noise_data * scale
        ann['audio']['array'] = conv
        ann['audio']['sampling_rate'] = self.sr
        return ann

    def _utts_list(self, speaker_dir: str):
        utts = []
        for idx, (dir_path, _, file_names) in enumerate(os.walk(speaker_dir)):
            if idx == 0:
                continue
            utts += file_names
        return utts

    def _shuffle_weight(self):
        return 0.45

def make_mixture(dataset: Callable, 
                 factor: int,
                 speech_dir: str,
                 ann_dir: str):
    if not os.path.exists(speech_dir):
        os.makedirs(speech_dir)
    if not os.path.exists(ann_dir):
        os.makedirs(ann_dir)
    for i in range(factor):
        for idx, sample in enumerate(dataset):
            rttm = []
            sr = sample['audio']['sampling_rate']
            speakers = np.unique(sample['speakers']).tolist()
            meeting = '-'.join(speakers) + '-' + f'{i}'  + '-' f'{idx}'
            for st, ed, speaker in zip(sample['st'], sample['ed'], sample['speakers']):
                rttm.append(f'SPEAKER {meeting} 1 {st/sr:.7} {(ed-st)/sr:.7} <NA> <NA> {speaker} <NA> <NA>')
            rttm = '\n'.join(rttm)
            with open(os.path.join(ann_dir, meeting + '.rttm'), 'w') as f:
                f.write(rttm)
            sf.write(os.path.join(speech_dir, meeting + '.wav'), sample['audio']['array'], sr, subtype='PCM_16')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type = int, default = 0)
    parser.add_argument("--speaker_root", type = str)
    parser.add_argument("--rir_root", type = str)
    parser.add_argument("--musan_root", type = str)
    parser.add_argument("--spk_dur_stat", type = str, default = "stat\\ch_speaker_dur.npy")
    parser.add_argument("--same_spk_pause_stat", type = str, default = "stat\\ch_same_spk_pause.npy")
    parser.add_argument("--diff_spk_pause_stat", type = str, default = "stat\\ch_diff_spk_pause.npy")
    parser.add_argument("--diff_spk_overlap_stat", type = str, default = "stat\\ch_diff_spk_overlap.npy")
    parser.add_argument("--Numin", type = int, default = 10)
    parser.add_argument("--Numax", type = int, default = 20)
    parser.add_argument("--maxlen", type = int, default = 3*60*16000)
    parser.add_argument("--n_speakers", type = int, default = 2)
    parser.add_argument("--factor", type = int, default = 1)
    parser.add_argument("--speech_dir", type = str, default = "DiLibriSpeech")
    parser.add_argument("--annotation_dir", type = str, default ="rttms")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    dataset = DiLibriSpeech(speaker_root = args.speaker_root,
                             rir_roots = [args.rir_root],
                             musan_root = args.musan_root,
                             spk_dur_stat = args.spk_dur_stat,
                             same_spk_pause_stat = args.same_spk_pause_stat,
                             diff_spk_pause_stat = args.diff_spk_pause_stat,
                             diff_spk_overlap_stat = args.diff_spk_overlap_stat,
                             Numin = args.Numin,
                             Numax = args.Numax,
                             maxlen = args.maxlen,
                             Nspeakers = args.n_speakers)
    make_mixture(dataset, 
                 factor = args.factor,
                 speech_dir = args.speech_dir,
                 ann_dir = args.annotation_dir)
