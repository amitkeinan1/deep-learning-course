import torch
from torch import nn
from tqdm import tqdm
from matplotlib import pyplot as plt

from ex3.gan.gan_get_data import get_encoded_mnist
from ex3.gan.gan_models import Generator, Discriminator
from ex3.gan.test_gan import show_latents_images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODING_DIM = 32
NOISE_DIM = ENCODING_DIM
GEN_LEARNING_RATE = 0.001
DIS_LEARNING_RATE = 0.0001
EPOCHS_NUM = 5


def train_gan():
    train_data, test_data = get_encoded_mnist()

    generator_losses = []
    discriminator_losses = []

    print("train gan")

    generator = Generator(input_dim=NOISE_DIM, output_dim=ENCODING_DIM)
    discriminator = Discriminator(input_dim=ENCODING_DIM)

    criterion = nn.BCELoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=GEN_LEARNING_RATE)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=DIS_LEARNING_RATE)

    for i in tqdm(range(EPOCHS_NUM)):
        print(f"epoch {i + 1} from {EPOCHS_NUM} started")

        for real_enc_images in train_data:
            batch_size = real_enc_images.shape[0]

            # train generator

            generator_optimizer.zero_grad()

            noise = torch.normal(mean=0.0, std=1.0, size=(batch_size, NOISE_DIM))  # TODO: make size batch X dim
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

        show_latents_images(generated_enc_images.select(0, 1).reshape((1, 32)))

    plt.plot(generator_losses, color='r')
    plt.plot(discriminator_losses, color='b')
    plt.show()


if __name__ == '__main__':
    train_gan()
