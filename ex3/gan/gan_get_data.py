import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from tqdm import tqdm

from ex3.ae.ae_models import Autoencoder
from ex3.gan.config import SAVE_DIR
from ex3.gan.encoded_images_dataset import GeneralDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


def get_mnist_data():
    print("get mnist data")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(2)])
    train_dataset = datasets.MNIST(
        root="~/torch_datasets", train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader


def load_pretrained_ae():
    print("get pretrained auto encoder")

    latent_dim = 32
    ae_path = os.path.join(SAVE_DIR, "ae_models",
                           "AE_model_latent_dim_32_lr_0.001_BatchSize_64_Epochs_20_2021-12-27_1113",
                           "best_Autoencoder.pth")
    model = Autoencoder(latent_dim=latent_dim).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(ae_path), DEVICE))

    model.eval()

    return model


def list_to_data_loader(encoded_images):
    encoded_images_data_set = GeneralDataset(encoded_images)
    encoded_images_data_loader = DataLoader(encoded_images_data_set, batch_size=BATCH_SIZE, shuffle=True)
    return encoded_images_data_loader


def encode_images(mnist_data_loader, auto_encoder):
    print("encoding images")

    all_encoded_images = []
    for data in tqdm(mnist_data_loader):
        curr_images, _ = data
        curr_images = curr_images.to(DEVICE)
        curr_encoded_images = auto_encoder.encoder(curr_images)
        all_encoded_images += list(curr_encoded_images)
    all_encoded_images = list_to_data_loader(all_encoded_images)
    return all_encoded_images


def save_object(object, path):
    with open(path, 'wb') as outp:
        pickle.dump(object, outp, pickle.HIGHEST_PROTOCOL)


def load_object(path):
    with open(path, 'rb') as inp:
        object = pickle.load(inp)
    return object


def get_encoded_mnist(reload=True):
    train_pickle_path = f"{SAVE_DIR}/encoded_images/encoded_train_images.pkl"
    test_pickle_path = f"{SAVE_DIR}/encoded_images/encoded_test_images.pkl"
    if reload:
        encoded_train_images = load_object(train_pickle_path)
        encoded_test_images = load_object(test_pickle_path)
    else:
        ae = load_pretrained_ae()
        train_loader, test_loader = get_mnist_data()
        encoded_train_images = encode_images(train_loader, ae)
        encoded_test_images = encode_images(test_loader, ae)
        save_object(encoded_train_images, train_pickle_path)
        save_object(encoded_test_images, test_pickle_path)

    return encoded_train_images, encoded_test_images


if __name__ == '__main__':
    encoded_train_images, encoded_test_images = get_encoded_mnist(reload=False)
