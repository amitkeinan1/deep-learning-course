from torch import nn

from data_handling import get_data, SEQUENCE_LEN, VOCABULARY
from evaluation import evaluate
from ff_net import FFNet
from training import train


def main():
    train_loader, test_loader = get_data()

    ff_net = FFNet(input_size=SEQUENCE_LEN * len(VOCABULARY))
    print(ff_net)
    criterion = nn.BCELoss()
    lr = 0.001
    train(model=ff_net, train_loader=train_loader, epochs_num=2, criterion=criterion, lr=lr)

    accuracy, recall, precision, f1 = evaluate(model=ff_net, test_loader=test_loader)


if __name__ == '__main__':
    main()
