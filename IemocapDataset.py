import os
import warnings

import torch
import torchaudio

import pandas as pd

class IemocapDataset(object):
    """
        Create a Dataset for Iemocap. Each item is a tuple of the form:
        (waveform, sample_rate, label)
    """

    _ext_audio = ".wav"
    _emotions = {'ang': 1, 'hap': 2, 'exc': 3, 'sad': 4, 'fru': 5, 'fea': 6, 'sur': 7, 'neu': 8 }

    def __init__(self, root):
        """
        Args:
            root_dir (string): Root directory containing the five session folders
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root

        # Iterate through all 5 sessions
        data = []
        for i in range(1, 6):
            # Define path of the session
            path = os.path.join(root, 'Session' + str(i), 'dialog', 'EmoEvaluation')

            # Get list of evaluation files
            files = [file for file in os.listdir(path) if file.endswith('.txt')]

            # Get utterance-level data from evaluation files
            # Iterate through evaluation files
            for file in files:
                # Open file
                f = open(os.path.join(path, file), 'r')

                # Get list of lines containing utterance-level data and split to individual string elements
                data += [line.strip()
                             .replace('[', '')
                             .replace(']', '')
                             .replace(' - ', '\t')
                             .replace(', ', '\t')
                             .split('\t')
                         for line in f if line.startswith('[') and not line.__contains__('xxx')]

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['start', 'end', 'file', 'emotion', 'activation', 'valence', 'dominance'], dtype=float)

        # Map emotion labels to numeric values
        self.df['emotion'] = self.df['emotion'].map(self._emotions)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root, self.df.loc[idx, 'file'])
        waveform, sample_rate = torchaudio.load(audio_name)
        label = self.df.loc[idx, 'emotion']

        sample = {'waveform': waveform, 'sample_rate': sample_rate, 'label': label}

        return sample