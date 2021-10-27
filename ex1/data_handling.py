import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from my_dataset import MyDataset

POS_SAMPLES_PATH = "data\\neg_A0201.txt"
NEG_SAMPLES_PATH = "data\\pos_A0201.txt"
SEQUENCE_LEN = 9
VOCABULARY = ['E', 'C', 'I', 'L', 'R', 'A', 'P', 'T', 'G', 'D', 'V', 'K', 'M', 'N', 'S', 'Q', 'W', 'H', 'Y', 'F']


def read_lines(file_path):
    with open(file_path, 'r') as input_file:
        lines = input_file.readlines()
    lines = [line.strip() for line in lines]
    return lines


def load_samples(pos_path, neg_path):
    pos_samples = read_lines(pos_path)
    neg_samples = read_lines(neg_path)
    samples = pos_samples + neg_samples
    labels = [1.0 for _ in range(len(pos_samples))] + [0.0 for _ in range(len(neg_samples))]
    return samples, labels


def array_from_nested_list(my_list):
    flat_list = [item for sublist in my_list for item in sublist]
    return np.array(flat_list)


def encode_seq(sequence, vocabulary):
    encodings = [[0.0 if char != letter else 1 for char in vocabulary] for letter in sequence]
    flat_array = array_from_nested_list(encodings)
    return flat_array


def split_and_encode(samples, labels, vocabulary):
    # encode
    samples = np.array([encode_seq(sample, vocabulary) for sample in samples])
    labels = np.array([np.array([label]) for label in labels])

    # split
    X_train, X_test, y_train, y_test = train_test_split(samples, labels, test_size=0.1)

    return X_train, X_test, y_train, y_test


def float_tensor_from_array(array):
    return torch.tensor(array).float()


def get_data():
    samples, labels = load_samples(POS_SAMPLES_PATH, NEG_SAMPLES_PATH)
    X_train, X_test, y_train, y_test = split_and_encode(samples, labels, VOCABULARY)

    train_set = MyDataset(float_tensor_from_array(X_train), float_tensor_from_array(y_train))
    train_loader = torch.utils.data.DataLoader(train_set)

    test_set = MyDataset(float_tensor_from_array(X_test), float_tensor_from_array(y_test))
    test_loader = torch.utils.data.DataLoader(test_set)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = get_data()
