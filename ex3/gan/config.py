import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODING_DIM = 32
NOISE_DIM = ENCODING_DIM
GEN_LEARNING_RATE = 0.005
DIS_LEARNING_RATE = 0.001
EPOCHS_NUM = 1

SAVE_DIR = "save_stuff"

