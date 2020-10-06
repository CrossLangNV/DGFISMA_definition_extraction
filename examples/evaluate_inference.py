"""
By Laurens
"""

import os
import time

from sklearn.metrics import classification_report

from data.load import get_gold_standard, get_training_data
from evaluate.predicting import predict_fasttext
from models.preconfigured import BERTForDefinitionClassification

ROOT = os.path.join(os.path.dirname(__file__), '..')


def main():
    """
    Evaluate BERT and fasttext model speed
    """

    if 0:
        sentences, labels = get_gold_standard()
    else:
        # Larger dataset to have a better indication of inference speed.
        sentences, labels = get_training_data()

    if 1:
        """
        BERT:
        """

        model_dir = os.path.join(ROOT, 'bert_classifier/models_dgfisma_def_extraction/retraining_example')

        # Load model
        # Ignore loading model time.
        model = BERTForDefinitionClassification.from_dir(model_dir)

        t0 = time.time()
        # Prediction

        pred_bert, _ = model.predict(sentences)

        t1 = time.time()
        t_bert = t1 - t0

        print('\033[1m', end='')
        print("BERT")
        print('\033[0m', end='')
        print(f"\tSpeed: T = {t_bert} s")
        classification_report(labels, pred_bert)

    """
    Fast text:
    """

    model_fasttext = os.path.join(ROOT, 'tests/test_files/arne/model_fasttext.bin')
    assert os.path.exists(model_fasttext)

    t0 = time.time()
    prediction_fasttext = predict_fasttext(model_fasttext, list(sentences))
    t1 = time.time()

    t_fasttext = t1 - t0

    print('\033[1m', end='')
    print("Fast text")
    print('\033[0m', end='')
    print(f"\tSpeed: T = {t_fasttext} s")
    classification_report(labels, prediction_fasttext)

    return


if __name__ == '__main__':
    main()
