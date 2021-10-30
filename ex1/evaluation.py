import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix


def get_labels_and_preds(model, test_loader):
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            preds = torch.round(outputs)

            all_labels.append(labels[0][0].item())
            all_preds.append(preds[0][0].item())
    model.train()

    return all_labels, all_preds


def evaluate(model, test_loader, verbose=True):
    labels, preds = get_labels_and_preds(model, test_loader)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    accuracy = accuracy_score(labels, preds)
    recall = recall_score(labels, preds)
    precision = precision_score(labels, preds)
    f1 = f1_score(labels, preds)

    if verbose:
        print(f"tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")
        print(f"accuracy: {accuracy}")
        print(f"recall: {recall}")
        print(f"precision: {precision}")
        print(f"f1: {f1}")

    return accuracy, recall, precision, f1


