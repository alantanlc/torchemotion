from IemocapDataset import *
import torch

def my_collate(batch):
    # data_1 = batch[0]['waveform'][:, 0:400]
    data = [item['waveform'][:,:400] for item in batch]
    target = [item['emotion'] for item in batch]
    return [data, target]

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load dataset
iemocap_dataset = IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release')

# split the dataset in train and test set
indices = torch.randperm(len(iemocap_dataset)).tolist()
dataset_train = torch.utils.data.Subset(iemocap_dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(iemocap_dataset, indices[-50:])

# define training and validation data loaders
data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=my_collate)
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=2, shuffle=False, num_workers=4, collate_fn=my_collate)

for i, data in enumerate(data_loader_train):
    # get the inputs; data is a  [waveform, label]
    x = 0