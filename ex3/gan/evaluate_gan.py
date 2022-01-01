import os

import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

from ex3.gan.config import NOISE_DIM, SAVE_DIR, ENCODING_DIM
from ex3.gan.general_dataset import GeneralDataset
from ex3.gan.gan_get_data import load_pretrained_ae, get_mnist_data, get_encoded_mnist
from ex3.gan.gan_models import generate_noise, Generator


def ae_latents_to_images(auto_encoder, latents_loader):
    all_images = []

    auto_encoder.eval()

    with torch.no_grad():
        for latents in latents_loader:
            images = auto_encoder.decoder.forward(latents)
            all_images += list(images)

    auto_encoder.train()

    return images


def gan_latents_to_images(generator, ae, gan_latents_loader):
    all_images = []

    generator.eval()
    ae.eval()

    with torch.no_grad():
        for gan_latents in gan_latents_loader:
            ae_latents = generator.forward(gan_latents)
            images = ae.decoder.forward(ae_latents)
            all_images += list(images)

    generator.train()
    ae.train()

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


def show_ae_latents_images(latents, title):
    latents_loader = DataLoader(GeneralDataset(latents), batch_size=32)
    auto_encoder = load_pretrained_ae()
    images = ae_latents_to_images(auto_encoder, latents_loader)
    show_images(images, title)

def show_gan_latents_images(latents, title):
    latents_loader = DataLoader(GeneralDataset(latents), batch_size=32)
    generator = load_saved_gan_generator("final3")
    ae = load_pretrained_ae()
    images = gan_latents_to_images(generator, ae, latents_loader)
    show_images(images, title)

def evaluate_gan_generator(generator, title):
    generator.eval()

    with torch.no_grad():
        noise = generate_noise(10, NOISE_DIM)
        generated_enc_images = generator(noise)
        show_ae_latents_images(generated_enc_images, title)

    generator.train()


def load_saved_gan_generator(training_name):
    generator_path = os.path.join(SAVE_DIR, "gans", f"{training_name}-generator.model")
    generator = Generator(input_dim=NOISE_DIM, output_dim=ENCODING_DIM)
    generator.load_state_dict(torch.load(generator_path))
    return generator


def show_random_mnist_images():
    train_loader, test_loader = get_mnist_data()
    for images, labels in test_loader:
        show_images(images[:10], "mnist images")
        break


def show_random_ae_mnist_images():
    encoded_train_images, encoded_test_images = get_encoded_mnist()
    for images in encoded_test_images:
        show_ae_latents_images(images[:10], "mnist encoded and decoded")
        break


def show_gan_random_images(training_name):
    generator = load_saved_gan_generator(training_name)
    evaluate_gan_generator(generator, training_name)


if __name__ == '__main__':
    latents = torch.ones(size=(10, 32))
    show_gan_latents_images(latents, "")