import torch
import pandas as pd

from ex1.data_handling import encode_seq, VOCABULARY
from ex1.models import FFNet2Hidden
from ex1.main import NEURONS_IN_LAYERS

SEQ_LENGTH = 9


def get_peptides():
    spike_path = r"C:\Users\AMIT\Google Drive\cloud\אקדמיה\נוכחיים\DL\deep-learning-course\ex1\data_to_infer\spike.txt"
    with open(spike_path, 'r') as input_file:
        spike_seq = input_file.read()
    peptides = []
    for i in range(len(spike_seq) - SEQ_LENGTH + 1):
        peptides.append(spike_seq[i: i + SEQ_LENGTH])
    return peptides


def get_model():
    model_path = r"C:\Users\AMIT\Google Drive\cloud\אקדמיה\נוכחיים\DL\deep-learning-course\ex1\models\model1.torch"
    model = FFNet2Hidden(SEQ_LENGTH * 20, NEURONS_IN_LAYERS)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def infer(model, peptide):
    model_input = torch.tensor(encode_seq(peptide, VOCABULARY)).float()
    with torch.no_grad():
        output = model(model_input)
        pred = torch.round(output)
    return output.item(), pred.item()


if __name__ == '__main__':
    peptides = get_peptides()
    model = get_model()
    scores = []
    predictions = []
    for peptide in peptides:
        score, prediction = infer(model, peptide)
        scores.append(score)
        predictions.append(prediction)

    positive_predictions = [pred for pred in predictions if pred == 1]
    print(f"{len(positive_predictions)} positive predictions out of {len(predictions)} predictions")

    top_scores_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    most_detectable = [(peptides[i], scores[i]) for i in top_scores_idxs]
    print("most_detectable:")
    print(most_detectable)

    results_df = pd.DataFrame(list(zip(peptides, scores, predictions)), columns=["peptides", "scores", "predictions"])
    results_df.to_excel("inference_results.xlsx")