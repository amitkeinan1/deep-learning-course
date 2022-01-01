import random

from ex3.gan.config import NOISE_DIM
from ex3.gan.evaluate_gan import show_ae_latents_images, show_gan_latents_images
from ex3.gan.gan_get_data import get_encoded_mnist
from ex3.gan.gan_models import generate_noise


def interpolate_in_gan_space(title):
    start = generate_noise(1, NOISE_DIM)[0]
    end = generate_noise(1, NOISE_DIM)[0]

    path_length = 10
    path = [start + ((end - start) * i / path_length) for i in range(path_length + 1)]

    show_gan_latents_images(path, f"interpolate-gan-{title}")


def interpolate_in_ae_space(title):
    _, encoded_test_images = get_encoded_mnist()
    encoded_test_images = encoded_test_images.dataset
    start = random.choice(encoded_test_images)
    end = random.choice(encoded_test_images)

    path_length = 10
    path = [start + ((end - start) * i / path_length) for i in range(path_length + 1)]

    show_ae_latents_images(path, f"interpolate-ae-{title}")


if __name__ == '__main__':
    interpolate_in_ae_space("")
