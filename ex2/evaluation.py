import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import itertools

def get_labels_and_preds(model, test_loader):
    all_labels = []
    all_preds = []
    model_name = model.name()
    model.eval()

    with torch.no_grad():
        for labels, reviews, reviews_text in test_loader:  # getting training batches
            # Recurrent nets (RNN/GRU)
            num_words = reviews.shape[1]
            if model_name in ['RNN', 'GRU']:
                hidden_state = model.init_hidden(int(labels.shape[0]))

                for i in range(num_words):
                    output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE

            else:
                # Token-wise networks (MLP / MLP + Atten.)
                sub_score = []
                if model_name != 'MLP':
                    # MLP + atten
                    sub_score, atten_weights = model(reviews)
                else:
                    # MLP
                    sub_score = model(reviews)

                output = torch.mean(sub_score, 1)

            all_labels.extend(np.argmax(labels.to('cpu').numpy(), axis=1))
            all_preds.extend(np.argmax(output.to('cpu').numpy(), axis=1))
    model.train()

    return all_labels, all_preds

def evaluate_model(model, test_loader, verbose=True):
    labels, preds = get_labels_and_preds(model, test_loader)

    cm = confusion_matrix(labels, preds)
    np.set_printoptions(precision=2)
    figure = plot_confusion_matrix(cm, classes=['pos','neg'],
                          title=model.name() + '_Confusion matrix without normalization')
    plt.figure()
    figure_normalize = plot_confusion_matrix(cm, classes=['pos','neg'], normalize=True,
                          title=model.name() + '_Normalized confusion matrix')
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)

    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    assert recall == tpr

    if verbose:
        print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
        print(f"tp rate: {tpr}, tn rate: {tnr}")
        print(f"accuracy: {accuracy}")
        print(f"recall: {recall}")
        print(f"precision: {precision}")
        print(f"f1: {f1}")
        # plt.show()
    return accuracy, recall, precision, f1, tpr, tnr, figure, figure_normalize



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    The confusion matrix is saved to a file
    Based on tutorial http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    Args:
        cm(int64,array) - a calculated confusion matrix
        classes(list) - The classes of the model
        normalize(bool) - Should we normalize the confusion matrix or not
        title(str) - The title for the confusion matrix
        cmap(color_map) - The colormap used to map normalized data values to RGBA colors.
                           more details on https://matplotlib.org/api/cm_api.html
    Returns:
        None
    """
    font = {'weight': 'bold',
            'size': 15}

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    figure = plt.figure(figsize=(20, 10))  # A large size was defined in order to make the output image readable
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(-0.5, len(classes) - 1 + 0.5)
    plt.xticks(tick_marks, classes, rotation=45, fontsize='large', fontweight='bold')
    plt.yticks(tick_marks, classes, fontsize='large', fontweight='bold')
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontdict=font)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    return figure