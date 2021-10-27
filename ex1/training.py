import torch.optim as optim
import torch.nn as nn

from data_handling import SEQUENCE_LEN, VOCABULARY
from models import FFNet1Hidden


def train(model, train_loader, epochs_num, criterion, lr):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(epochs_num):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == '__main__':
    ff_net = FFNet1Hidden(input_size=SEQUENCE_LEN * len(VOCABULARY))
    print(ff_net)
    criterion = nn.BCELoss()
    lr = 0.001
    train(model=ff_net, epochs_num=2, criterion=criterion, lr=lr)
