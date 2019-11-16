from IemocapDataset import *

iemocap_dataset = IemocapDataset('/home/alanwuha/Documents/Projects/iemocap/IEMOCAP_full_release')

for i in range(len(iemocap_dataset)):
    sample = iemocap_dataset[i]
    print(sample)