import torch.nn as nn
import torch.nn.functional as F

class VGG_convnet(nn.Module):

    def __init__(self):

        super(VGG_convnet, self).__init__()

        # block 1:        1 x 64240 --> 64 x 200
        self.conv1a = nn.Conv1d(1, 64, kernel_size=400, stride=160)
        self.conv1b = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(2)

        # block 2:          64 x 200 --> 128 x 50
        self.conv2a = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv2b = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(4)

        # block 3:          128 x 50 --> 256 x 12
        self.conv3a = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv3b = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(4)

        # block 4:          256 x 12 --> 512 x 4
        self.conv4a = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool1d(3)

        # linear layers:    512 x 4 --> 2048 --> 4096 --> 4096 --> 9
        self.linear1 = nn.Linear(2048, 4096)
        self.linear2 = nn.Linear(4096, 4096)
        self.linear3 = nn.Linear(4096, 9)

    def forward(self, x):

        # block 1
        x = self.conv1a(x)
        x = F.relu(x)
        x = self.conv1b(x)
        x = F.relu(x)
        x = self.pool1(x)

        # block 2
        x = self.conv2a(x)
        x = F.relu(x)
        x = self.conv2b(x)
        x = F.relu(x)
        x = self.pool2(x)

        # block 3
        x = self.conv3a(x)
        x = F.relu(x)
        x = self.conv3b(x)
        x = F.relu(x)
        x = self.pool3(x)

        # block 4:
        x = self.conv4a(x)
        x = F.relu(x)
        x = self.pool4(x)

        # linear layers
        x = x.view(-1, 2048)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)

        return x