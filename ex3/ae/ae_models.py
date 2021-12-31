import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Loading sataset, use toy = True for obtaining a smaller dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


class Encoder(nn.Module):
    def __init__(self, latent_dim=10, last_filter=64):
        super(Encoder, self).__init__()
        self.conv_1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)
        self.ReLU = torch.nn.ReLU()
        self.conv_2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        last_filter = last_filter
        self.conv_3 = nn.Conv2d(32, last_filter, 3, stride=2, padding=1)
        self.flat = nn.Flatten()
        self.lest_latent_space = nn.Linear(last_filter * 4 * 4, latent_dim)

    def calc_out_conv_layers(self, in_h, in_w, kernels, paddings, dilations, strides):
        out_h = in_h
        out_w = in_w
        for ker, pad, dil, stri in zip(kernels, paddings, dilations, strides):
            out_h = (out_h + 2 * pad - dil * (ker - 1) - 1) / stri + 1
            out_w = (out_w + 2 * pad - dil * (ker - 1) - 1) / stri + 1

        return out_h, out_w

    def name(self):
        return "Encoder"

    def forward(self, x):
        # print('input AE: ', x.shape)
        x = self.conv_1(x)
        x = self.ReLU(x)
        # print('after conv_1: ', x.shape)

        x = self.conv_2(x)
        x = self.ReLU(x)
        # print('after conv_2: ', x.shape)

        x = self.conv_3(x)
        # print('after conv_3: ', x.shape)
        x = self.flat(x)
        # print('after flat: ', x.shape)
        x = self.lest_latent_space(x)
        # print('after lest_latent_space: ', x.shape)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=10):
        super(Decoder, self).__init__()
        self.lest_latent_space = nn.Linear(latent_dim, 16)
        self.Tconv_1 = nn.ConvTranspose2d(16, 32, 5, stride=2, padding=1, output_padding=1)
        self.ReLU = torch.nn.ReLU()
        self.Tconv_2 = nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1)
        self.Tconv_3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.Tconv_5 = nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1)
        self.sigmoid = nn.Sigmoid()

    def name(self):
        return "Decoder"

    def forward(self, x):
        x = self.lest_latent_space(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 1, 1))
        x = self.Tconv_1(x)
        x = self.ReLU(x)
        x = self.Tconv_2(x)
        x = self.ReLU(x)
        x = self.Tconv_3(x)
        x = self.ReLU(x)
        x = self.ReLU(x)
        x = self.Tconv_5(x)
        x = self.sigmoid(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=10, last_filter=64):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim=latent_dim, last_filter=last_filter)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.layers_list = torch.nn.ModuleList()
        self.layers_list.append(self.encoder)
        self.layers_list.append(self.decoder)

    def name(self):
        return "Autoencoder"

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class Classifier(nn.Module):
    def __init__(self, model_encoder=None, latent_dim=10, transfer_learning=True, last_filter=64):
        super(Autoencoder, self).__init__()
        if transfer_learning:
            if model_encoder is None:
                raise "did not give trained model for transfer_learning as var"
            self.encoder = model_encoder
            for i, param in enumerate(self.encoder.parameters()):
                if i == 0:
                    param.requires_grad = False
        else:
            self.encoder = Encoder(latent_dim=latent_dim, last_filter=last_filter)
        self.ReLU = torch.nn.ReLU()
        self.fc1 = nn.Linear(self.encoder.lest_latent_space.out_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)
        self.layers_list = torch.nn.ModuleList()
        self.layers_list.append(self.encoder)
        self.layers_list.append(self.fc1)
        self.layers_list.append(self.fc2)
        self.layers_list.append(self.fc3)

    def name(self):
        return "Autoencoder"

    def forward(self, x):
        x = self.encoder(x)
        x = self.ReLU(x)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        x = self.ReLU(x)
        return x
