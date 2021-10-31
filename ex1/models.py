import torch
import torch.nn as nn
import torch.nn.functional as F

from data_handling import VOCABULARY, SEQUENCE_LEN

INPUT_SIZE = SEQUENCE_LEN * len(VOCABULARY)


def generate_model(hidden_layers_num, neurons_in_hidden_layers):
    assert (len(neurons_in_hidden_layers) == hidden_layers_num)

    # if hidden_layers_num == 0:
    #     model = FFNet0Hidden(INPUT_SIZE, neurons_in_hidden_layers)

    if hidden_layers_num == 1:
        model = FFNet1Hidden(INPUT_SIZE, neurons_in_hidden_layers)

    # elif hidden_layers_num == 2:
    #     model = FFNet2Hidden(INPUT_SIZE, neurons_in_hidden_layers)
    #
    # elif hidden_layers_num == 5:
    #     model = FFNet5Hidden(INPUT_SIZE, neurons_in_hidden_layers)
    #
    # elif hidden_layers_num == 10:
    #     model = FFNet10Hidden(INPUT_SIZE, neurons_in_hidden_layers)

    else:
        raise Exception(f"we don't support {hidden_layers_num} hidden layers")

    return model


class FFNet0Hidden(nn.Module):

    def __init__(self, input_size, hidden_layers_sizes):
        super(FFNet0Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class FFNet1Hidden(nn.Module):

    def __init__(self, input_size, hidden_layers_sizes):
        super(FFNet1Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers_sizes[0])
        self.fc2 = nn.Linear(hidden_layers_sizes[0], 1)
        self.leaky_relu = torch.nn.Tanh()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x


class FFNet2Hidden(nn.Module):

    def __init__(self, input_size, hidden_layers_sizes):
        super(FFNet2Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers_sizes[0])
        self.fc2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
        self.fc3 = nn.Linear(hidden_layers_sizes[1], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class FFNet5Hidden(nn.Module):

    def __init__(self, input_size, hidden_layers_sizes):
        super(FFNet5Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers_sizes[0])
        self.fc2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
        self.fc3 = nn.Linear(hidden_layers_sizes[1], hidden_layers_sizes[2])
        self.fc4 = nn.Linear(hidden_layers_sizes[2], hidden_layers_sizes[3])
        self.fc5 = nn.Linear(hidden_layers_sizes[3], hidden_layers_sizes[4])
        self.fc6 = nn.Linear(hidden_layers_sizes[4], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = torch.sigmoid(x)
        return x


class FFNet10Hidden(nn.Module):

    def __init__(self, input_size, hidden_layers_sizes):
        super(FFNet10Hidden, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_layers_sizes[0])
        self.fc2 = nn.Linear(hidden_layers_sizes[0], hidden_layers_sizes[1])
        self.fc3 = nn.Linear(hidden_layers_sizes[1], hidden_layers_sizes[2])
        self.fc4 = nn.Linear(hidden_layers_sizes[2], hidden_layers_sizes[3])
        self.fc5 = nn.Linear(hidden_layers_sizes[3], hidden_layers_sizes[4])
        self.fc6 = nn.Linear(hidden_layers_sizes[4], hidden_layers_sizes[5])
        self.fc7 = nn.Linear(hidden_layers_sizes[5], hidden_layers_sizes[6])
        self.fc8 = nn.Linear(hidden_layers_sizes[6], hidden_layers_sizes[7])
        self.fc9 = nn.Linear(hidden_layers_sizes[7], hidden_layers_sizes[8])
        self.fc10 = nn.Linear(hidden_layers_sizes[8], hidden_layers_sizes[9])
        self.fc11 = nn.Linear(hidden_layers_sizes[9], 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = self.fc11(x)
        x = torch.sigmoid(x)
        return x
