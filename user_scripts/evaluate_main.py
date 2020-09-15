"""
By Laurens
"""

import os
import time

from bert_classifier.src.predict import DefExtractModel
from data.load import get_gold_standard
from evaluate.eval import report
from evaluate.predicting import predict_fasttext

ROOT = os.path.join(os.path.dirname(__file__), '..')


def main():
    """
    Evaluate BERT and fasttext model
    """

    sentences, labels = get_gold_standard()

    if 1:
        """
        BERT:
        """

        model_bert_path = os.path.join(ROOT,
                                       'bert_classifier/models_dgfisma_def_extraction/run_2020_06_26_11_56_31_acb319aac70b/distilbert-base-uncased_model_10.pth')

        # Load model
        model_bert = DefExtractModel(model_bert_path)

        t0 = time.time()
        # Prediction
        pred_bert, flat_predictions_proba = model_bert.predict(sentences)

        t1 = time.time()
        t_bert = t1 - t0

        print('\033[1m', end='')
        print("BERT")
        print('\033[0m', end='')
        print(f"\tSpeed: T = {t_bert} s")
        report(labels, pred_bert)

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
    report(labels, prediction_fasttext)

    return


if __name__ == '__main__':
    main()
