import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from ex3.gan.encoded_images_dataset import EncodedImagesDataset
from ex3.gan.gan_get_data import get_pretrained_ae


def latents_to_images(auto_encoder, latents_loader):
    all_images = []

    auto_encoder.eval()

    with torch.no_grad():
        for latents in latents_loader:
            images = auto_encoder.decoder.forward(latents)
            all_images += list(images)

    return images


def show_images(images):
    for image in images:
        plt.imshow(image[0])
        plt.show()


def show_latents_images(latents):
    latents_loader = DataLoader(EncodedImagesDataset(latents), batch_size=32)
    auto_encoder = get_pretrained_ae()
    images = latents_to_images(auto_encoder, latents_loader)
    show_images(images)


if __name__ == '__main__':
    fake_latents = [torch.normal(0, 0, size=(32,)) for i in range(32)]
    show_latents_images(fake_latents)
