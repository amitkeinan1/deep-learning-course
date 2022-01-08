import torch

from ex3.gan.config import NOISE_DIM, DIGITS_NUM
from ex3.gan.evaluate_gan import load_saved_gan_generator, show_ae_latents_images
from ex3.gan.gan_models import generate_noise
from ex3.gan.gan_utils import merge_tensor_and_labels


def main(training_name):
    generator = load_saved_gan_generator(training_name, conditional=True)

    generator.eval()

    with torch.no_grad():
        examples_num = 10
        noise = generate_noise(1, NOISE_DIM).repeat(examples_num, 1)
        # noise = torch.ones(1, NOISE_DIM).repeat(examples_num, 1)
        labels = torch.tensor(list(range(DIGITS_NUM)))
        gen_input = merge_tensor_and_labels(noise, labels)
        generated_enc_images = generator(gen_input)
        show_ae_latents_images(generated_enc_images, f"extra analysis - {training_name}")

    generator.train()


if __name__ == '__main__':
    main("conditional-15-21-33")
