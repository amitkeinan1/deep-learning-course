import os
import random
from datetime import datetime

import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np

from ex3.gan.config import NOISE_DIM, ENCODING_DIM, GEN_LEARNING_RATE, DIS_LEARNING_RATE, EPOCHS_NUM, SAVE_DIR, \
    DIGITS_NUM
from ex3.gan.gan_get_data import get_encoded_mnist
from ex3.gan.gan_models import Generator, Discriminator, generate_noise
from ex3.gan.evaluate_gan import evaluate_gan_generator
from ex3.gan.gan_utils import merge_tensor_and_labels

current_time = datetime.now().strftime("%H:%M:%S").replace(':', '-')

TRAINING_NAME = f"conditional-{current_time}"


def smooth_list(list1, n=1000):
    smoothed_list = [np.mean(list1[i - n: i + n]) for i in range(n, len(list1) - n)]
    return smoothed_list


def plot_losses(generator_losses, discriminator_losses, title, smoothed=False):
    if smoothed:
        generator_losses = smooth_list(generator_losses)
        discriminator_losses = smooth_list(discriminator_losses)
    plt.title(f"training losses - {title}")
    plt.plot(generator_losses, color='r', label="generator")
    plt.plot(discriminator_losses, color='b', label="discriminator")
    plt.legend()
    plt.savefig(f"{SAVE_DIR}/plots/{title}.png")
    plt.show()


def train_gan(conditional=False):
    train_data, test_data = get_encoded_mnist(reload=True, conditional=conditional)

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
            if conditional:
                real_enc_images, real_labels = real_enc_images

            batch_size = real_enc_images.shape[0]

            # train generator

            generator_optimizer.zero_grad()

            noise = generate_noise(batch_size, NOISE_DIM)
            if not conditional:
                generated_enc_images = generator(noise)
            else:
                random_labels = torch.tensor([random.randint(0, 9) for _ in range(batch_size)])
                noise_and_labels = merge_tensor_and_labels(noise, random_labels)
                generated_enc_images = generator(noise_and_labels)

            if not conditional:
                disc_input = generated_enc_images
            else:
                disc_input = merge_tensor_and_labels(generated_enc_images, random_labels)
            disc_out_on_gen = discriminator(disc_input)

            generator_loss = criterion(disc_out_on_gen, torch.ones(
                size=(batch_size, 1)))  # ones because we want the discriminator to classify as true
            generator_loss.backward()
            generator_optimizer.step()

            generator_losses.append(generator_loss.item())

            # train discriminator

            discriminator_optimizer.zero_grad()

            disc_out_on_gen = discriminator(disc_input.detach())  # detach from graph because now we train the disc.
            if not conditional:
                disc_out_on_real = discriminator(real_enc_images)
            else:
                real_images_and_labels = merge_tensor_and_labels(real_enc_images, real_labels)
                disc_out_on_real = discriminator(real_images_and_labels)
            disc_outputs = torch.cat([disc_out_on_gen, disc_out_on_real])
            disc_labels = torch.cat([torch.zeros(size=(batch_size, 1)),
                                     torch.ones(
                                         size=(batch_size, 1))])  # real labels - zeros for generated, ones for real

            discriminator_loss = criterion(disc_outputs, disc_labels)
            discriminator_loss.backward()
            discriminator_optimizer.step()

            discriminator_losses.append(discriminator_loss.item())

        title = f"{TRAINING_NAME} - epoch {epoch_num}"
        plot_losses(generator_losses, discriminator_losses, title, smoothed=True)
        evaluate_gan_generator(generator, title, conditional)

    title = f"{TRAINING_NAME} - final"
    plot_losses(generator_losses, discriminator_losses, f"final - {TRAINING_NAME}", smoothed=True)
    evaluate_gan_generator(generator, title, conditional)
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, "gans", f"{TRAINING_NAME}-generator.model"))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, "gans", f"{TRAINING_NAME}-discriminator.model"))


if __name__ == '__main__':
    train_gan(conditional=True)
