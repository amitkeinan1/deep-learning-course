import os
from sentiment_start import ExMLP, ExLRestSelfAtten, ExRNN, ExGRU
import torch
import torch.nn as nn
import numpy as np
import loader as ld
import matplotlib.pyplot as plt
from evaluation import evaluate_model
import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

def plot_losses(train_losses, test_losses, batch_counts, train_name):
    figure = plt.figure()
    plt.plot(batch_counts, train_losses, 'o--', label="train", alpha=0.8)
    plt.plot(batch_counts, test_losses, 'o--', label="test", alpha=0.8)

    plt.title(f"train and test losses - {train_name}")
    plt.xlabel("batches count")
    plt.ylabel("loss per item")
    plt.legend()
    # plt.savefig(os.path.join(FIGS_PATH, str(random.random()) + ".png"))
    # plt.show()
    return figure


def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    text = rev_text[:20] if len(rev_text) > 20 else rev_text
    finale_text = ''
    for i in range(len(text)):
        word = text[i]
        finale_text += '"' + word + '"' + ' scores: [' + str(sbs1[i]) + ',' + str(sbs2[i]) + '] \n'
    final_predict_1 = np.mean(sbs1)
    final_predict_2 = np.mean(sbs2)
    finale_text += 'final_predict: [' + str(final_predict_1) + ',' + str(final_predict_2) + ']\n'
    finale_text += 'true score: [' + str(lbl1) + ',' + str(lbl2) + ']\n'
    print(finale_text)


def train_network(model_name,
                  output_size,
                  hidden_size,
                  output_dir='',
                  num_epochs=10,
                  batch_size=32,
                  atten_size=0,
                  reload_model=False,
                  learning_rate=0.001,
                  test_interval=100,
                  load_model_path='',
                  Best=False,

                  ):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)
    if model_name in ['RNN', 'GRU']:
        if model_name in ['RNN']:
            model = ExRNN(input_size, output_size, hidden_size).to(device)
        else:
            model = ExGRU(input_size, output_size, hidden_size).to(device)
    else:
        if atten_size > 0 and model_name in ["MLP_atten"]:
            # pass
            model = ExLRestSelfAtten(input_size, output_size, hidden_size, atten_size).to(device)
        else:
            model = ExMLP(input_size, output_size, hidden_size).to(device)

    print("Using model: " + model.name())
    print(model)

    if reload_model:
        if len(load_model_path):
            print("Reloading model")
            tmp_model_name = "best_" + model.name() if Best else model.name()
            model.load_state_dict(torch.load(os.path.join(load_model_path, tmp_model_name + ".pth")))
        else:
            print('did not give path to model')

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loss = 1.0
    test_loss = 1.0

    # training steps in which a test step is executed every test_interval
    train_loss_list = []
    test_loss_list = []
    itrrations = []
    best_accuracy = 0
    best_recall = 0
    best_precision = 0
    best_f1 = 0
    model_name = model.name()
    model_path = ''

    if len(output_dir):
        now = datetime.datetime.now()
        time_stamp = now.strftime("%Y-%m-%d_%H%M")
        final_name = '_'.join((model_name, f"lr_{learning_rate}", f"HiddSize_{hidden_size}", f"AttSize_{atten_size}",
                               f"BatchSize_{batch_size}", f"Epochs_{num_epochs}", time_stamp))
        model_path = os.path.join(output_dir, final_name)
        os.makedirs(model_path, exist_ok=True)
    for epoch in range(num_epochs):
        itr = 0  # iteration counter within each epoch

        for labels, reviews, reviews_text in train_dataset:  # getting training batches

            itr = itr + 1

            if (itr + 1) % test_interval == 0:
                test_iter = True
                labels, reviews, reviews_text = next(iter(test_dataset))  # get a test batch
            else:
                test_iter = False

            # Recurrent nets (RNN/GRU)

            if model.name() in ['RNN', 'GRU']:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:, i, :].to(device), hidden_state.to(device))  # HIDE

            else:

                # Token-wise networks (MLP / MLP + Atten.)

                if model_name == "MLP_atten":
                    # MLP + atten
                    sub_scores, atten_weights = model(reviews)
                else:  # MLP
                    assert model_name == "MLP"
                    sub_scores = model(reviews)

                means = sub_scores.mean(axis=1)
                softmax = torch.nn.Softmax(dim=1)
                output = softmax(means)

            # cross-entropy loss

            loss = criterion(output, labels)

            # optimize in training iterations

            if not test_iter:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # averaged losses

            test_loss = 0.8 * float(loss.detach()) + 0.2 * test_loss

            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

            if test_iter:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], "
                    f"Step [{itr + 1}/{len(train_dataset)}], "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Test Loss: {test_loss:.4f}"
                )

                # saving the model
                torch.save(model, os.path.join(model_path, model.name() + ".pth"))

        if model.name() not in ['RNN', 'GRU']:
            nump_subs = sub_scores.to('cpu').detach().numpy()
            labels = labels.to('cpu').detach().numpy()
            # print_review(reviews_text[0], nump_subs[0, :, 0], nump_subs[0, :, 1], labels[0, 0], labels[0, 1])

        print("********")
        print("test metrics:")
        accuracy, recall, precision, f1, tpr, tnr, figure, figure_normalize = evaluate_model(model, test_dataset,
                                                                                             verbose=True)

        figure.savefig(os.path.join(model_path, model.name() + "_Confusion_matrix_without_normalization" + ".png"))
        figure_normalize.savefig(
            os.path.join(model_path, model.name() + "_Confusion_matrix_with_normalization" + ".png"))
        itrrations.append(epoch)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)
        if f1 >= best_f1:
            best_accuracy, best_recall, best_precision, best_f1 = (accuracy, recall, precision, f1)

            torch.save(model, os.path.join(model_path, "best_" + model.name() + ".pth"))
            figure.savefig(
                os.path.join(model_path, model.name() + "_Best_Confusion_matrix_without_normalization" + ".png"))
            figure_normalize.savefig(
                os.path.join(model_path, model.name() + "_Best_Confusion_matrix_with_normalization" + ".png"))
        plt.close(figure)
        plt.close(figure_normalize)

    test_final_metrics = evaluate_model(model, test_dataset, verbose=False)
    train_final_metrics = evaluate_model(model, train_dataset, verbose=False)

    figure = plot_losses(train_loss_list, test_loss_list, itrrations, '')
    figure.savefig(os.path.join(model_path, model.name() + "_TrainAndTest_Loss_Plot" + ".png"))
    # return best_f1, best_accuracy, best_recall, best_precision
    return test_final_metrics, train_final_metrics


def main():
    batch_size = [16, 32, 64, 128]
    output_size = 2
    hidden_size = [64, 80, 96, 112, 128]  # to experiment with

    atten_size = [1, 3, 5, 7]  # need atten > 0 for using restricted self atten
    reload_model = False
    num_epochs = 1  # 30
    learning_rate = [0.001, 0.005, 0.01]
    test_interval = 100
    output_dir = 'models'
    att_size = 0
    for model_name in ['MLP', 'MLP_atten', 'GRU', 'RNN']:
        for lr in learning_rate:
            for b_s in batch_size:
                for h_size in hidden_size:
                    if model_name == 'MLP_atten':
                        for att_size in atten_size:
                            best_f1, best_accuracy, best_recall, best_precision = train_network(model_name,
                                                                                                output_size,
                                                                                                h_size,
                                                                                                num_epochs=num_epochs,
                                                                                                batch_size=b_s,
                                                                                                atten_size=att_size,
                                                                                                reload_model=reload_model,
                                                                                                learning_rate=lr,
                                                                                                test_interval=test_interval,
                                                                                                output_dir=output_dir)
                    else:
                        best_f1, best_accuracy, best_recall, best_precision = train_network(model_name,
                                                                                            output_size,
                                                                                            h_size,
                                                                                            num_epochs=num_epochs,
                                                                                            batch_size=b_s,
                                                                                            atten_size=att_size,
                                                                                            reload_model=reload_model,
                                                                                            learning_rate=lr,
                                                                                            test_interval=test_interval,
                                                                                            output_dir=output_dir)
        print(model_name + ': ', best_f1, best_accuracy, best_recall, best_precision)


if __name__ == '__main__':
    main()
