from torch import nn

from data_handling import get_data, SEQUENCE_LEN, VOCABULARY
from evaluation import evaluate
from models import generate_model
from training import train


def run_single_training(train_loader, test_loader, hidden_layers_num, neurons_in_hidden_layers, activation_func):
    model = generate_model(hidden_layers_num, neurons_in_hidden_layers, activation_func)

    criterion = nn.BCELoss()
    lr = 0.001
    train(model=model, train_loader=train_loader, epochs_num=2, criterion=criterion, lr=lr)

    accuracy, recall, precision, f1 = evaluate(model=model, test_loader=test_loader)
    return accuracy


def main():
    train_loader, test_loader = get_data()
    accuracy = run_single_training(train_loader, test_loader, hidden_layers_num=2, neurons_in_hidden_layers=[32, 32],
                                   activation_func=None)
    print(accuracy)


if __name__ == '__main__':
    main()
