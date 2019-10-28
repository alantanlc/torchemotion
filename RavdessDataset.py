from __future__ import print_function, division
import os
import torch
import pandas as pd
from torch.utils.data import Dataset
import torchaudio

class RavdessDataset(Dataset):
    """RavdessDataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.ravdess_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ravdess_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = os.path.join(self.root_dir, self.ravdess_frame.iloc[idx, 0])
        waveform, sample_rate = torchaudio.load(filename)
        label = self.ravdess_frame.iloc[idx, 1]

        if self.transform:
            waveform = self.transform(waveform)

        sample = {'waveform': waveform, 'label': label}

        return sample
