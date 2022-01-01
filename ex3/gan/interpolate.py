from ex3.gan.config import NOISE_DIM
from ex3.gan.evaluate_gan import show_ae_latents_images, show_gan_latents_images
from ex3.gan.gan_models import generate_noise


def interpolate_in_gan_space(title):
    start = generate_noise(1, NOISE_DIM)[0]
    end = generate_noise(1, NOISE_DIM)[0]

    path_length = 10
    path = [start + ((end - start) * i / path_length) for i in range(path_length + 1)]

    show_gan_latents_images(path, f"interpolate-gan-{i}")
    

if __name__ == '__main__':
    for i in range(20):
        interpolate_in_gan_space(i)
