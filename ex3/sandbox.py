import os

import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

from ae_models import Autoencoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    train_dataset = datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    # train_dataset = list(train_dataset)[:4096]
    test_dataset = datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=8)
    return train_loader, test_loader


def get_pretrained_ae():
    latent_dim = 32
    ae_path = os.path.join("ae_models", "AE_model_latent_dim_32_lr_0.001_BatchSize_64_Epochs_20_2021-12-27_1113",
                           "best_Autoencoder.pth")
    model = Autoencoder(latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(ae_path), DEVICE))
    return model


if __name__ == '__main__':

    train_loader, test_loader = get_mnist_data()
    ae = get_pretrained_ae()

    for data in test_loader:
        images, _ = data
        # print(images.shape)
        plt.imshow(images[0].reshape(32, 32))
        plt.show()
        # break
        images = images.to(DEVICE)
        encoded_images = ae.encoder(images)
        print(encoded_images.shape)
        # print(latent_space)
