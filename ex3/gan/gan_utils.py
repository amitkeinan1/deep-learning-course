import torch
import torch.nn.functional as F

from ex3.gan.config import DIGITS_NUM


def merge_tensor_and_labels(tensor, labels):
    one_hot_labels = F.one_hot(labels, DIGITS_NUM)
    tensor_and_labels = torch.cat((tensor, one_hot_labels), dim=1)
    return tensor_and_labels
