import os

import torch
from torchvision import datasets, transforms
from matplotlib import pyplot as plt

from ae_models import Autoencoder

if __name__ == '__main__':
    a = torch.normal(mean=0.0, std=1.0, size=[1,10])
    print(a)

    # train_loader, test_loader = get_mnist_data()
    # ae = get_pretrained_ae()
    #
    # for data in test_loader:
    #     images, _ = data
    #     # print(images.shape)
    #     plt.imshow(images[0].reshape(32, 32))
    #     plt.show()
    #     # break
    #     images = images.to(DEVICE)
    #     encoded_images = ae.encoder(images)
    #     print(encoded_images.shape)
    #     # print(latent_space)
