import os
import torch
import torchaudio
import pandas as pd
import numpy as np

class EmodbDataset(object):
    """
        Create a Dataset for Emodb. Each item is a tuple of the form:
        (waveform, sample_rate, emotion)
    """

    _ext_audio = '.wav'
    _emotions = { 'W': 1, 'L': 2, 'E': 3, 'A': 4, 'F': 5, 'T': 6, 'N': 7 } # W = anger, L = boredom, E = disgust, A = anxiety/fear, F = happiness, T = sadness, N = neutral

    def __init__(self, root='download'):
        """
        Args:
            root (string): Directory containing the wav folder
        """
        self.root = root

        # Iterate through all audio files
        data = []
        for _, _, files in os.walk(root):
            for file in files:
                if file.endswith(self._ext_audio):
                    # Construct file identifiers
                    identifiers = [file[0:2], file[2:5], file[5], file[6], os.path.join('wav', file)]

                    # Append identifier to data
                    data.append(identifiers)

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['speaker_id', 'code', 'emotion', 'version', 'file'], dtype=np.float32)

        # Map emotion labels to numeric values
        self.df['emotion'] = self.df['emotion'].map(self._emotions).astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root, self.df.loc[idx, 'file'])
        waveform, sample_rate = torchaudio.load(audio_name)
        emotion = self.df.loc[idx, 'emotion']

        sample = {
            'waveform': waveform,
            'sample_rate': sample_rate,
            'emotion': emotion
        }

        return sample

# Example: Load Emodb dataset
# emodb_dataset = EmodbDataset('/home/alanwuha/Documents/Projects/datasets/emodb/download')

# Example: Iterate through samples
# for i in range(len(emodb_dataset)):
#     sample = emodb_dataset[i]
#     print(i, sample)