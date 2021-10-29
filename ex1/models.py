import torch
import torch.nn as nn
import torch.nn.functional as F

from data_handling import VOCABULARY, SEQUENCE_LEN

INPUT_SIZE = SEQUENCE_LEN * len(VOCABULARY)


def generate_model(hidden_layers_num, neurons_in_hidden_layers):
    assert(len(neurons_in_hidden_layers) == hidden_layers_num)

    if hidden_layers_num == 0:
        model = FFNet0Hidden(INPUT_SIZE, neurons_in_hidden_layers)

    elif hidden_layers_num == 1:
        model = FFNet1Hidden(INPUT_SIZE, neurons_in_hidden_layers)

    elif hidden_layers_num == 2:
        model = FFNet2Hidden(INPUT_SIZE, neurons_in_hidden_layers)

    else:
        raise Exception("we don't support it")

    return model


# class FFNet0Hidden(nn.Module):
#
#     def __init__(self, input_size, hidden_layers_sizes):
#         super(FFNet0Hidden, self).__init__()
#         self.fc1 = nn.Linear(input_size, 1)
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = torch.sigmoid(x)
#         return x


class FFNet1Hidden(nn.Module):

    def __init__(self, input_size, hidden_layers_sizes):
        super(FFNet1Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers_sizes[0])
        self.fc2 = nn.Linear(hidden_layers_sizes[0], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


# class FFNet2Hidden(nn.Module):
#
#     def __init__(self, input_size, hidden_layers_sizes):
#         super(FFNet2Hidden, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_layers_sizes[0])
#         self.fc2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
#         self.fc3 = nn.Linear(hidden_layers_sizes[1], 1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         x = torch.sigmoid(x)
#         return x
