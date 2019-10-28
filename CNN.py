import torch.nn as nn
import torch

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # block 1:          1 x 32 x 129 --> 32 x 16 x 65
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)

        # block 2:          32 x 16 x 65 --> 64 x 8 x 33
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(2, 2)

        # block 3:          64 x 8 x 33 --> 16384 --> 1024
        self.fc = nn.Linear(16384, 1024)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        # block 1:          1 x 32 x 129 --> 32 x 16 x 64
        x = self.conv1(x)
        x = self.pool1(x)

        # block 2:          32 x 16 x 64 --> 64 x 8 x 32
        x = self.conv2(x)
        x = self.pool2(x)

        # block 3:          64 x 8 x 32 --> 16384 --> 1024
        x = x.view(-1, 16384)
        x = self.fc(x)
        x = self.dropout(x)

        return x

# Test
# x = torch.rand(1, 1, 32, 129)
# net = CNN()
# y = net(x)
# print(y)