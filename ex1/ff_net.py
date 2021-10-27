import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNet(nn.Module):

    def __init__(self, input_size):
        super(FFNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
