from torch import nn

from data_handling import get_data, SEQUENCE_LEN, VOCABULARY
from evaluation import evaluate
from models import generate_model
from training import train

BATCH_SIZE = 64
LEARNING_RATE = 0.002
EPOCHS_NUMBER = 25


def lr_search(learning_rates):
    accuracies = []
    for lr in learning_rates:
        accuracy = run_single_training(hidden_layers_num=1, neurons_in_hidden_layers=[256], activation_func=None,
                                       epochs_num=20, batch_size=BATCH_SIZE, lr=lr, criterion=nn.BCELoss(),
                                       train_name=f"learning rate: {lr}")
    accuracies.append(accuracy)
    print({lr: accuracy for lr, accuracy in zip(learning_rates, accuracies)})


def architectural_params_search():
    results = []
    layers_num_and_neurons_nums = {0: [[]],
                                   1: [[8 * (2 ** i)] for i in range(6)],
                                   2: [[8 * (2 ** i), 8 * (2 ** i)] for i in range(6)]}

    for hidden_layers_num, neurons_nums_list in layers_num_and_neurons_nums.items():
        for neurons_nums in neurons_nums_list:
            print("***")
            print(hidden_layers_num, neurons_nums)
            accuracy = run_single_training(hidden_layers_num, neurons_nums, None, 2)
            results.append((hidden_layers_num, neurons_nums, accuracy))

    print(results)


def run_single_training(hidden_layers_num, neurons_in_hidden_layers,
                        criterion, train_name, epochs_num=EPOCHS_NUMBER, batch_size=BATCH_SIZE, lr=LEARNING_RATE):
    train_loader, test_loader = get_data(batch_size)
    model = generate_model(hidden_layers_num, neurons_in_hidden_layers)
    print(model)

    train(model=model, train_loader=train_loader, test_loader=test_loader, epochs_num=epochs_num, criterion=criterion,
          lr=lr, train_name=train_name)

    accuracy, recall, precision, f1 = evaluate(model=model, test_loader=test_loader)
    return accuracy


def one_train():
    accuracy = run_single_training(hidden_layers_num=1, neurons_in_hidden_layers=[128], criterion=nn.BCELoss(),
                                   train_name="")
    print(accuracy)


if __name__ == '__main__':
    one_train()
