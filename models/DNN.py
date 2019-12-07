import torch.nn as nn
import torch.nn.functional as F
import torch

class DNN(nn.Module):

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DNN, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size1, bias = False)
        self.linear2 = nn.Linear(hidden_size1, hidden_size2, bias = False)
        self.linear3 = nn.Linear(hidden_size2, output_size, bias = False)

    def forward(self, x):

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        scores = self.linear3(x)

        return scores

# Test
# x = torch.rand(384)
# net = DNN(384, 512, 512, 512)
# y = net(x)
# print(y)