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
    def __init__(self, input_length: int):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(int(input_length), 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear1(x))


class GAN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generator = Generator(input_dim, output_dim)
        self.discriminator = Discriminator(input_dim, output_dim)

    def forward(self, x, batch_size):
        random_noise = torch.normal(mean=0.0, std=1.0, size=[batch_size, self.input_dim])
        generated_vector = self.generator.forward(random_noise)





