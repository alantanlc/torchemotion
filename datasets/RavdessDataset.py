import os
import torch
import torchaudio
import pandas as pd
import numpy as np

class RavdessDataset(object):
    """
        Create a Dataset for Ravdess. Each item is a tuple of the form:
        (waveform, sample_rate, emotion)
    """

    _ext_audio = '.wav'
    _emotions = { 'neu': 1, 'cal': 2, 'hap': 3, 'sad': 4, 'ang': 5, 'fea': 6, 'dis': 7, 'sur': 8 }

    def __init__(self, root='Audio_Speech_Actors_01-24'):
        """
        Args:
            root (string): Directory containing the Actor folders
        """
        self.root = root

        # Iterate through all audio files
        data = []
        for _, _, files in os.walk(root):
            for file in files:
                if file.endswith(self._ext_audio):
                    # Truncate file extension and split filename identifiers
                    identifiers = file[:-len(self._ext_audio)].split('-')

                    # Append file path w.r.t to root directory
                    identifiers.append(os.path.join('Actor_' + identifiers[-1], file))

                    # Append identifier to data
                    data.append(identifiers)

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['modality', 'vocal_channel', 'emotion', 'intensity', 'statement', 'repetition', 'actor', 'file'], dtype=np.float32)

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

# Example: Load Ravdess dataset
# ravdess_dataset = RavdessDataset('/home/alanwuha/Documents/Projects/datasets/ravdess/Audio_Speech_Actors_01-24')

# Example: Iterate through samples
# for i in range(len(ravdess_dataset)):
#     sample = ravdess_dataset[i]
#     print(i, sample)