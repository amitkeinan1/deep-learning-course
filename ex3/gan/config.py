import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODING_DIM = 32
NOISE_DIM = ENCODING_DIM
GEN_LEARNING_RATE = 0.001
DIS_LEARNING_RATE = 0.001
EPOCHS_NUM = 35

DIGITS_NUM = 10

SAVE_DIR = "save_stuff"

