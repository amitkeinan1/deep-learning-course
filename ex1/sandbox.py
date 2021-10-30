import torch
from torch import nn

if __name__ == '__main__':
    labels = torch.tensor([1.0, 1.0])
    outputs = torch.tensor([0.5, 0.9])
    criterion = nn.BCELoss()
    print(criterion(outputs, labels))
