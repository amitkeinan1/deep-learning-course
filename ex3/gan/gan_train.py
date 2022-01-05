import os

import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from ex3.gan.config import NOISE_DIM, ENCODING_DIM, GEN_LEARNING_RATE, DIS_LEARNING_RATE, EPOCHS_NUM, SAVE_DIR, \
    DIGITS_NUM
from ex3.gan.gan_get_data import get_encoded_mnist
from ex3.gan.gan_models import Generator, Discriminator, generate_noise
from ex3.gan.evaluate_gan import evaluate_gan_generator

TRAINING_NAME = "final3"


def plot_losses(generator_losses, discriminator_losses, title):
    plt.title(f"training losses - {title}")
    plt.plot(generator_losses, color='r', label="generator")
    plt.plot(discriminator_losses, color='b', label="discriminator")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/plots/{title}.png")
    plt.show()


def train_gan(conditional=False):
    train_data, test_data = get_encoded_mnist(conditional)

    generator_losses = []
    discriminator_losses = []

    print("train gan")

    if not conditional:
        generator = Generator(input_dim=NOISE_DIM, output_dim=ENCODING_DIM)
        discriminator = Discriminator(input_dim=ENCODING_DIM)
    else:
        generator = Generator(input_dim=NOISE_DIM + DIGITS_NUM, output_dim=ENCODING_DIM)
        discriminator = Discriminator(input_dim=ENCODING_DIM + DIGITS_NUM)

    criterion = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=GEN_LEARNING_RATE)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=DIS_LEARNING_RATE)

    for epoch_num in tqdm(range(EPOCHS_NUM)):
        print(f"epoch {epoch_num + 1} from {EPOCHS_NUM} started")

        for real_enc_images in train_data:
            batch_size = real_enc_images.shape[0]

            # train generator

            generator_optimizer.zero_grad()

            noise = generate_noise(batch_size, NOISE_DIM)
            generated_enc_images = generator(noise)

            disc_out_on_gen = discriminator(generated_enc_images)

            generator_loss = criterion(disc_out_on_gen, torch.ones(
                size=(batch_size, 1)))  # ones because we want the discriminator to classify as true
            generator_loss.backward()
            generator_optimizer.step()

            generator_losses.append(generator_loss.item())

            # train discriminator

            discriminator_optimizer.zero_grad()

            disc_out_on_gen = discriminator(
                generated_enc_images.detach())  # detach from graph because now we train the disc.
            disc_out_on_real = discriminator(real_enc_images)
            disc_outputs = torch.cat([disc_out_on_gen, disc_out_on_real])
            labels = torch.cat([torch.zeros(size=(batch_size, 1)),
                                torch.ones(size=(batch_size, 1))])  # real labels - zeros for generated, ones for real

            discriminator_loss = criterion(disc_outputs, labels)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            discriminator_losses.append(discriminator_loss.item())

        title = f"{TRAINING_NAME} - epoch {epoch_num}"
        plot_losses(generator_losses, discriminator_losses, title)
        evaluate_gan_generator(generator, title)

    title = f"{TRAINING_NAME} - final"
    plot_losses(generator_losses, discriminator_losses, f"final - {TRAINING_NAME}")
    evaluate_gan_generator(generator, title)
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, "gans", f"{TRAINING_NAME}-generator.model"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, "gans", f"{TRAINING_NAME}-discriminator.model"))


if __name__ == '__main__':
    train_gan()
