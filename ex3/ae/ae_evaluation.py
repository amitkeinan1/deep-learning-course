import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report

def get_labels_and_preds(model, test_loader,device):
    all_labels = []
    all_preds = []
    model_name = model.name()
    model.eval()

    with torch.no_grad():
        for img, labels in test_loader:  # getting training batches
            # Recurrent nets (RNN/GRU)
            num_words = img.shape[0]
            img = img.to(device)
            labels = labels.to(device)
            output = model(img)

            all_labels.extend(labels.to('cpu').numpy())
            all_preds.extend(output.to('cpu').numpy())

    return all_labels, all_preds

def evaluate_model(model, test_loader,device, verbose=True):
    figure = ''
    figure_normalize = ''
    labels, preds = get_labels_and_preds(model, test_loader,device)
    predicted_class = np.argmax(preds,axis=1)
    true_classes_binarized = label_binarize(labels, classes=list(i for i in range(10)))
    predicted_classes_binarized = label_binarize(predicted_class, classes=list(i for i in range(10)))
    cm = confusion_matrix(labels, predicted_class)
    np.set_printoptions(precision=2)
    if verbose:
        figure = plot_confusion_matrix(cm, classes=list(str(i) for i in range(10)),
                              title=model.name() + '_Confusion matrix without normalization')
        plt.figure()
        figure_normalize = plot_confusion_matrix(cm, classes=list(str(i) for i in range(10)), normalize=True,
                              title=model.name() + '_Normalized confusion matrix')
    accuracy = accuracy_score(true_classes_binarized, predicted_classes_binarized,normalize=True)
    recall = recall_score(true_classes_binarized, predicted_classes_binarized,average = 'weighted', zero_division = 0)
    precision = precision_score(true_classes_binarized, predicted_classes_binarized,average = 'weighted', zero_division = 0)
    f1 = f1_score(true_classes_binarized, predicted_classes_binarized,average = 'weighted', zero_division = 0)
    report = classification_report(labels, predicted_class, target_names=list(str(i) for i in range(10)), zero_division = 0)


    print("The accuracy of the model is: " + str(accuracy) + '\n')
    print(f"accuracy: {accuracy}")
    print(f"recall: {recall}")
    print(f"precision: {precision}")
    print(f"f1: {f1}")
    return accuracy, recall, precision, f1, figure, figure_normalize, report



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
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis])
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
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