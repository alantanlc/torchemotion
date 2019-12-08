import os
import time
import torch
import torchaudio
from datasets.IemocapDataset import IemocapDataset

# Log start time
start_time = time.time()

# Specify file path and name (include file extension)
config_file_path = 'config/IS09_emotion.conf'
output_file_name = 'iemocap_is09_emotion.csv'

# Load dataset and construct dataloader
dataset = IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=IemocapDataset.collate_fn_segments)

# Remove output file if exists
if os.path.exists(output_file_name):
    os.system('rm ' + output_file_name)

# Initialize stft spectrogram transform
specgram_transform = torchaudio.transforms.Spectrogram(n_fft=256, win_length=256, hop_length=128)

# Iterate through dataset using dataloader to get segments of each utterance
for segments, emotions, n_segments, filenames in dataloader:
    # For each segment,
    # 1. Extract 384d IS09 Emotion features using SMILExtract
    # 2. Extract 32x129 spectogram using torchaudio transform (16ms frame size, 8ms step size, 256 fft bins)
    for i in range(len(segments)):
        # 1. Save segment as .wav file and extract 384d IS09 emotion features using SMILExtract
        torchaudio.save('segment.wav', segments[i], sample_rate=16000, precision=16, channels_first=False)
        execute_string = 'SMILExtract -C ' + config_file_path + ' -I ' + 'segment.wav' + ' -csvoutput ' + output_file_name + ' -instname ' + str(i) + '_' + filenames[i]
        os.system(execute_string)

        # 2. Extract 32x129 spectrograms using torchaudio transform (16ms frame size, 8ms step size, 256 fft bins)
        specgram = specgram_transform(segments[i].view(1, -1))

# Iterate through dataset and extract features of each sample using SMILExtract
# For os.system(execute_string) to work on Linux:
# 1. Ensure that SMILExtract executable exists in /usr/local/bin.
# 2. Ensure that SMILExtract executable can be executed without sudo.
#    - This can be achieved by granting all users with the permission to execute the SMILExtract executable using the following command: chmod a+x /usr/local/bin/SMILExtract
# for i in range(len(dataset)):
#     execute_string = 'SMILExtract -C ' + config_file_path + ' -I ' + dataset[i]['path'] + ' -csvoutput ' + output_file_name
#     os.system(execute_string)

# Compute and print program execution time
end_time = time.time()
total_time = end_time - start_time
print('Program took %d min %d sec to complete' % (total_time // 60, total_time % 60))