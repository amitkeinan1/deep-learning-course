from torch import nn
import gc

from data_handling import get_data, SEQUENCE_LEN, VOCABULARY
from evaluation import evaluate
from models import generate_model
from training import train

BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS_NUMBER = 40

CRITERION = nn.BCELoss()
HIDDEN_LAYERS_NUM = 2
NEURONS_IN_LAYERS = [128, 128]


def lr_search(learning_rates=None):
    if learning_rates is None:
        learning_rates = [10 ** (-4), 5 * 10 ** (-4), 10 ** (-3), 5 * 10 ** (-3), 10 ** (-2), 5 * 10 ** (-2),
                          10 ** (-1)]
    accuracies = []
    for lr in learning_rates:
        accuracy = run_single_training(hidden_layers_num=1, neurons_in_hidden_layers=[256],
                                       epochs_num=40, batch_size=BATCH_SIZE, lr=lr, criterion=nn.BCELoss(),
                                       train_name=f"learning rate: {lr}")
        accuracies.append(accuracy)
        print(f"temp accuracies: {accuracies}")
        gc.collect()
    print({lr: accuracy for lr, accuracy in zip(learning_rates, accuracies)})


def loss_search(loss_functions):
    accuracies = []
    for loss in loss_functions:
        accuracy = run_single_training(hidden_layers_num=1, neurons_in_hidden_layers=[256], criterion=loss,
                                       train_name=f"loss function: {loss}")
    accuracies.append(accuracy)
    print({loss: accuracy for loss, accuracy in zip(loss_functions, accuracies)})


def architectural_params_search():
    results = []
    layers_num_and_neurons_nums = {0: [[]],
                                   1: [[32], [64], [128], [256], [512], [1024], [2048]],
                                   2: [[16] * 2, [32] * 2, [64] * 2, [128] * 2, [256] * 2, [512] * 2, [1024] * 2],
                                   5: [[128, 128, 128, 128, 128]]}

    # layers_num_and_neurons_nums = {i: [[256] * i] * 3 for i in [2, 5, 10]}

    for hidden_layers_num, neurons_nums_list in layers_num_and_neurons_nums.items():
        for neurons_nums in neurons_nums_list:
            print("***")
            print(hidden_layers_num, neurons_nums)
            accuracy = run_single_training(hidden_layers_num=hidden_layers_num, neurons_in_hidden_layers=neurons_nums,
                                           train_name="")
            results.append((hidden_layers_num, neurons_nums, accuracy))

        print("temp results")
        print(results)

    print("final results")
    print(results)


def run_single_training(train_name="", hidden_layers_num=HIDDEN_LAYERS_NUM, neurons_in_hidden_layers=NEURONS_IN_LAYERS,
                        criterion=CRITERION,
                        epochs_num=EPOCHS_NUMBER, batch_size=BATCH_SIZE,
                        lr=LEARNING_RATE):
    train_loader, test_loader = get_data(batch_size)
    model = generate_model(hidden_layers_num, neurons_in_hidden_layers)
    print(model)

    train(model=model, train_loader=train_loader, test_loader=test_loader, epochs_num=epochs_num, criterion=criterion,
          lr=lr, train_name=train_name)

    print("test metrics")
    accuracy, recall, precision, f1, tpr, tnr = evaluate(model=model, test_loader=test_loader)

    print("train metrics")
    accuracy, recall, precision, f1, tpr, tnr = evaluate(model=model, test_loader=train_loader)

    return accuracy


if __name__ == '__main__':
    run_single_training()
