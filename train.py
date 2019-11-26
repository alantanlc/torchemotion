from IemocapDataset import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import time
import copy

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # Set model to training mode
            else:
                model.eval()    # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, emotions in dataloaders[phase]:
                inputs = inputs.to(device)
                emotions = emotions.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, emotions)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0) / 1
                running_corrects += torch.sum(preds == emotions.data) / 1

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss
            epoch_acc = running_corrects.double()

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# load dataset
iemocap_dataset = IemocapDataset('/home/alanwuha/Documents/Projects/datasets/iemocap/IEMOCAP_full_release')

# split the dataset in train and test set
indices = torch.randperm(len(iemocap_dataset)).tolist()
datasets = {
    'train': torch.utils.data.Subset(iemocap_dataset, indices[:-50]),
    'val': torch.utils.data.Subset(iemocap_dataset, indices[-50:])
}
dataloaders = { x: torch.utils.data.DataLoader(datasets[x], batch_size=4, shuffle=True, num_workers=4, collate_fn=IemocapDataset.collate_fn) for x in ['train', 'val'] }