from torch import nn

from data_handling import get_data, SEQUENCE_LEN, VOCABULARY
from evaluation import evaluate
from models import generate_model
from training import train


def run_single_training(hidden_layers_num, neurons_in_hidden_layers, activation_func, epochs_num):
    train_loader, test_loader = get_data()
    model = generate_model(hidden_layers_num, neurons_in_hidden_layers, activation_func)
    print(model)

    criterion = nn.BCELoss()
    lr = 0.001
    train(model=model, train_loader=train_loader, test_loader=test_loader, epochs_num=epochs_num, criterion=criterion, lr=lr)

    accuracy, recall, precision, f1 = evaluate(model=model, test_loader=test_loader)
    return accuracy


def main():
    accuracy = run_single_training(hidden_layers_num=1, neurons_in_hidden_layers=[128], activation_func=None, epochs_num=5)
    print(accuracy)


if __name__ == '__main__':
    main()
