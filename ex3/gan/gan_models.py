import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(int(input_dim), int(output_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear1(x))


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(int(input_dim), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear1(x))






