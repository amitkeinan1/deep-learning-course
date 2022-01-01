import os

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from ex3.gan.config import NOISE_DIM, SAVE_DIR, ENCODING_DIM
from ex3.gan.encoded_images_dataset import EncodedImagesDataset
from ex3.gan.gan_get_data import get_pretrained_ae
from ex3.gan.gan_models import generate_noise, Generator


def latents_to_images(auto_encoder, latents_loader):
    all_images = []

    auto_encoder.eval()

    with torch.no_grad():
        for latents in latents_loader:
            images = auto_encoder.decoder.forward(latents)
            all_images += list(images)

    return images


def show_images(images, title):
    figure = plt.figure(figsize=(len(images), 1))
    plt.title(title)
    for i, image in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(image[0])
        plt.axis('off')
    plt.savefig(f"{SAVE_DIR}/images/{title}.png")
    plt.show()
    plt.close(figure)


def show_latents_images(latents, title):
    latents_loader = DataLoader(EncodedImagesDataset(latents), batch_size=32)
    auto_encoder = get_pretrained_ae()
    images = latents_to_images(auto_encoder, latents_loader)
    show_images(images, title)


def evaluate_gan_generator(generator, title):
    generator.eval()

    with torch.no_grad():
        noise = generate_noise(10, NOISE_DIM)
        generated_enc_images = generator(noise)
        show_latents_images(generated_enc_images, title)

    generator.train()


def load_saved_gan_generator(training_name):
    generator_path = os.path.join(SAVE_DIR, "gans", f"{training_name}-generator.model")
    generator = Generator(input_dim=NOISE_DIM, output_dim=ENCODING_DIM)
    generator.load_state_dict(torch.load(generator_path))
    return generator


if __name__ == '__main__':
    fake_latents = [torch.normal(0, 0, size=(32,)) for i in range(5)]
    show_latents_images(fake_latents)
