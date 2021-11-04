import torch
import numpy as np

from ex1.inference import get_model

SEQUENCE_LEN = 9
VOCABULARY = ['E', 'C', 'I', 'L', 'R', 'A', 'P', 'T', 'G', 'D', 'V', 'K', 'M', 'N', 'S', 'Q', 'W', 'H', 'Y', 'F']
VOCABULARY_SIZE = len(VOCABULARY)


def tensor_to_peptide(tensor):
    list1 = tensor.tolist()
    max_idxs = []

    for i in range(SEQUENCE_LEN):
        start_idx = i * VOCABULARY_SIZE
        end_idx = start_idx + VOCABULARY_SIZE
        max_idx = np.argmin(list1[start_idx: end_idx])
        max_idxs.append(max_idx)

    peptide = [VOCABULARY[idx] for idx in max_idxs]

    return ''.join(peptide)


def optimize_sequence():
    model = get_model()
    model.requires_grad_(False)
    input = torch.nn.Parameter(torch.nn.Parameter(torch.randn(SEQUENCE_LEN * VOCABULARY_SIZE), requires_grad=True))
    optim = torch.optim.SGD([input], lr=10 ** (-2))
    y = torch.zeros(1)  # the desired network response

    criterion = torch.nn.BCELoss()

    num_steps = 100  # how many optim steps to take
    for _ in range(num_steps):
        print(input.tolist())
        loss = criterion(model(input), y)
        print(loss)
        loss.backward()
        optim.step()
        optim.zero_grad()

    peptide = tensor_to_peptide(input)

    return peptide


if __name__ == '__main__':
    peptide = optimize_sequence()
    print(peptide)
