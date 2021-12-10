import os
import torch
from torch.utils.data import DataLoader

from experiments import MODELS_OUTPUT_DIR
from loader import ReviewDataset, collact_batch

BATCH_SIZE = 32


def load_model(model_type, model_name):
    path = os.path.join(MODELS_OUTPUT_DIR, model_name, model_type + ".pth")
    model = torch.load(path)
    model.eval()
    return model


def get_processed_reviews():
    reviews = {"I liked this movie a lot, this is the best movie this year!" : "pos",
               "I did not like this movie, this is not the best movie this year...": "neg"}
    test_data = ReviewDataset(list(reviews.keys()), list(reviews.values()))
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collact_batch)
    return test_loader, reviews


if __name__ == '__main__':
    model = load_model("MLP_atten", "single_mlp_attention_documented")
    processed_reviews, textual_reviews = get_processed_reviews()

    preds = []
    for labels, reviews, reviews_text in processed_reviews:
        sub_scores, attention_weights = model(reviews)
        for sent_sub_scores, review_text in zip(sub_scores, reviews_text):
            for sub_score, word in zip(sent_sub_scores, review_text):
                print(word)
                print(sub_score)
                print()

        means = sub_scores.mean(axis=1)
        softmax = torch.nn.Softmax(dim=1)
        outputs = softmax(means)
        for output in outputs:
            print(float(output[0]))
            is_positive = (round(float(output[0])) == 1)
            preds.append(is_positive)

    for review, pred in zip(textual_reviews, preds):
        print(review)
        print(pred)
        print()
