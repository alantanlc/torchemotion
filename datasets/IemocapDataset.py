import os
import torch
import torchaudio
import pandas as pd
import numpy as np
import torch.nn.functional as F

class IemocapDataset(object):
    """
        Create a Dataset for Iemocap. Each item is a tuple of the form:
        (waveform, sample_rate, emotion, activation, valence, dominance)
    """

    _ext_audio = '.wav'
    _emotions = { 'ang': 0, 'hap': 1, 'exc': 1, 'sad': 3, 'fru': 4, 'fea': 5, 'sur': 6, 'neu': 7, 'xxx': 8 }

    def __init__(self,
                 root='IEMOCAP_full_release',
                 emotions=['ang', 'hap', 'exc', 'sad', 'neu'],
                 sessions=[1, 2, 3, 4, 5],
                 script_impro=['script', 'impro'],
                 genders=['M', 'F']):
        """
        Args:
            root (string): Directory containing the Session folders
        """
        self.root = root

        # Iterate through all 5 sessions
        data = []
        for i in range(1, 6):
            # Define path to evaluation files of this session
            path = os.path.join(root, 'Session' + str(i), 'dialog', 'EmoEvaluation')

            # Get list of evaluation files
            files = [file for file in os.listdir(path) if file.endswith('.txt')]

            # Iterate through evaluation files to get utterance-level data
            for file in files:
                # Open file
                f = open(os.path.join(path, file), 'r')

                # Get list of lines containing utterance-level data. Trim and split each line into individual string elements.
                data += [line.strip()
                             .replace('[', '')
                             .replace(']', '')
                             .replace(' - ', '\t')
                             .replace(', ', '\t')
                             .split('\t')
                         for line in f if line.startswith('[')]

        # Get session number, script/impro, speaker gender, utterance number
        data = [d + [d[2][4], d[2].split('_')[1], d[2][-4], d[2][-3:]] for d in data]

        # Create pandas dataframe
        self.df = pd.DataFrame(data, columns=['start', 'end', 'file', 'emotion', 'activation', 'valence', 'dominance', 'session', 'script_impro', 'gender', 'utterance'], dtype=np.float32)

        # Filter by emotions
        filtered_emotions = self.df['emotion'].isin(emotions)
        self.df = self.df[filtered_emotions]

        # Filter by sessions
        filtered_sessions = self.df['session'].isin(sessions)
        self.df = self.df[filtered_sessions]

        # Filter by script_impro
        filtered_script_impro = self.df['script_impro'].str.contains('|'.join(script_impro))
        self.df = self.df[filtered_script_impro]

        # Filter by gender
        filtered_genders = self.df['gender'].isin(genders)
        self.df = self.df[filtered_genders]

        # Reset indices
        self.df = self.df.reset_index()

        # Map emotion labels to numeric values
        self.df['emotion'] = self.df['emotion'].map(self._emotions).astype(np.float32)

        # Map file to correct path w.r.t to root
        self.df['file'] = [os.path.join('Session' + file[4], 'sentences', 'wav', file[:-5], file + self._ext_audio) for file in self.df['file']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_name = os.path.join(self.root, self.df.loc[idx, 'file'])
        waveform, sample_rate = torchaudio.load(audio_name)
        emotion = self.df.loc[idx, 'emotion']
        activation = self.df.loc[idx, 'activation']
        valence = self.df.loc[idx, 'valence']
        dominance = self.df.loc[idx, 'dominance']

        sample = {
            'path': audio_name,
            'waveform': waveform,
            'sample_rate': sample_rate,
            'emotion': emotion,
            'activation': activation,
            'valence': valence,
            'dominance': dominance
        }

        return sample

    def collage_fn_vgg(batch):
        # Clip or pad each utterance audio into 4.020 seconds.
        sample_rate = 16000
        n_channels = 1
        frame_length = np.int(4.020 * sample_rate)

        # Initialize output
        waveforms = torch.zeros(0, n_channels, frame_length)
        emotions = torch.zeros(0)

        for item in batch:
            waveform = item['waveform']
            original_waveform_length = waveform.shape[1]
            padded_waveform = F.pad(waveform, (0, frame_length - original_waveform_length)) if original_waveform_length < frame_length else waveform[:, :frame_length]
            waveforms = torch.cat((waveforms, padded_waveform.unsqueeze(0)))
            emotions = torch.cat((emotions, torch.tensor([item['emotion']])), 0)

        return waveforms, emotions

    def collate_fn_segments(batch):
        sample_rate = 16000
        segment_length = np.int(0.264 * sample_rate)
        step_length = np.int(0.025 * sample_rate)

        # Initialize output
        segments = torch.zeros(0, segment_length)
        n_segments = torch.zeros(0)
        emotions = torch.zeros(0)
        filenames = []

        # Iterate through samples in batch
        for item in batch:
            waveform = item['waveform']
            original_waveform_length = waveform.shape[1]

            # Compute number of segments given input waveform, segment, and step lengths
            item_n_segments = np.int(np.ceil((original_waveform_length - segment_length) / step_length) + 1)

            # Compute and apply padding to waveform
            padding_length = segment_length - original_waveform_length if original_waveform_length < segment_length else (segment_length + (item_n_segments - 1) * step_length - original_waveform_length)
            padded_waveform = F.pad(waveform, (0, padding_length))
            padded_waveform = padded_waveform.view(-1)

            # Construct tensor of segments
            item_segments = torch.zeros(item_n_segments, segment_length)
            for i in range(item_n_segments):
                item_segments[i] = padded_waveform[i*step_length:i*step_length+segment_length]
            segments = torch.cat((segments, item_segments), 0)

            # Construct tensor of emotion labels
            emotion = torch.tensor([item['emotion']])
            emotions = torch.cat((emotions, emotion.repeat(item_n_segments)), 0)

            # Construct list of
            filenames += [item['path'].split('/')[-1]]*item_n_segments

            # Construct tensor of n_frames (contains a list of number of frames per item)
            item_n_segments = torch.tensor([float(item_n_segments)])
            n_segments = torch.cat((n_segments, item_n_segments), 0)

        return segments, emotions, n_segments, filenames

    def collate_fn(batch):
        # Frame the signal into 20-40ms frames. 25ms is standard.
        # This means that the frame length for a 16kHz signal is 0.025 * 16000 = 400 samples.
        # Frame step is usually something like 10ms (160 samples), which allows some overlap to the frames.
        # The first 400 sample frame starts at sample 0, the next 400 sample frame starts at sample 160 etc until the end of the speech file is reached.
        # If the speech file does not divide into an even number, pad it with zeros so that it does.
        sample_rate = 16000
        # n_channels = 1
        frame_length = np.int(0.025 * sample_rate)
        step_length = np.int(0.01 * sample_rate)

        # Initialize output
        # frames = torch.zeros(0, n_channels, frame_length)
        frames = torch.zeros(0, frame_length)
        n_frames = torch.zeros(0)
        emotions = torch.zeros(0)

        for item in batch:
            waveform = item['waveform']
            original_waveform_length = waveform.shape[1]

            # Compute number of frames given input waveform, frame and step lengths
            item_n_frames = np.int(np.ceil((original_waveform_length - frame_length) / step_length) + 1)

            # Compute and apply padding to waveform
            padding_length = frame_length - original_waveform_length if original_waveform_length < frame_length else (frame_length + (item_n_frames - 1) * step_length - original_waveform_length)
            padded_waveform = F.pad(waveform, (0, padding_length))
            padded_waveform = padded_waveform.view(-1)

            # Construct tensor of frames
            # item_frames = torch.zeros(n_frames, n_channels, frame_length)
            item_frames = torch.zeros(item_n_frames, frame_length)
            for i in range(item_n_frames):
                item_frames[i] = padded_waveform[i*step_length:i*step_length+frame_length]
                # item_frames[i] = padded_waveform[:, i*step_length:i*step_length+frame_length]
            frames = torch.cat((frames, item_frames), 0)

            # Construct tensor of emotion labels
            emotion = torch.tensor([item['emotion']])
            emotions = torch.cat((emotions, emotion.repeat(item_n_frames)), 0)

            # Construct tensor of n_frames (contains a list of number of frames per item)
            item_n_frames = torch.tensor([float(item_n_frames)])
            n_frames = torch.cat((n_frames, item_n_frames), 0)

        return frames, emotions, n_frames

# Example: Load Iemocap dataset
# iemocap_dataset = IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release')

# Example: Iterate through samples
# for i in range(len(iemocap_dataset)):
#     sample = iemocap_dataset[i]
#     print(i, sample)

# Number of audio by duration
# dataset_duration = np.ceil(iemocap_dataset.df['end'] - iemocap_dataset.df['start'])
# idx = np.where(dataset_duration == 35)
# durations = np.unique(dataset_duration)
# durations_count = [np.sum(dataset_duration == i) for i in durations]

# print('End')