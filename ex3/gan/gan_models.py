import torch
import torch.nn as nn


class Generator(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.linear1 = nn.Linear(int(input_dim), 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, int(output_dim))

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(int(input_dim), 64)
        self.linear2 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        return x


def generate_noise(batch_size, noise_dim):
    noise = torch.normal(mean=0.0, std=1.0, size=(batch_size, noise_dim))
    return noise
