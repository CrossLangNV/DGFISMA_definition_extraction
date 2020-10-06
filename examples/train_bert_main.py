"""
* Load data
* Split data
* Init a model
* Train
* Evaluate
"""
import os

import pandas as pd
from numpy import random
from sklearn.metrics import classification_report

from data.examples import data_main
from models.preconfigured import BERTForDefinitionClassification

DELIMITER = "￭"


def main(seed,
         folder_out='.',
         val_ratio=.2):
    # Data preparation
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_main(seed,
                                                                         val_ratio=val_ratio)

    model_storage_directory = 'bert_classifier_wordpiece/models_dgfisma_def_extraction'
    model = BERTForDefinitionClassification.from_distilbert(model_storage_directory)

    model.train(x_train, y_train, validation_data=(x_valid, y_valid), epochs=10)

    pred_test, _ = model.predict(x_test)

    model_info = {}

    n_train = len(x_train)

    def get_data_info(n_train,
                      report,
                      label,
                      dataset):

        labels = report.keys()
        assert label in labels

        info = {'set': dataset,
                'n_train': n_train,
                'label': label,
                }

        info.update(report[label])

        return info

    l = []

    for dataset, x, y, pred in [
        # ('train', x_train, y_train, pred_train),
        # ('validation', x_valid, y_valid, pred_valid),
        ('test', x_test, y_test, pred_test)
    ]:
        report = classification_report(y, pred, output_dict=True)

        for label in ['0', '1', 'weighted avg']:
            info = get_data_info(n_train,
                                 report,
                                 label,
                                 dataset)

            info.update(model_info)

            l.append(info)

    df = pd.DataFrame(l)

    filename_save = os.path.join(folder_out, f'eval_s{seed}.csv')
    if not os.path.exists(os.path.dirname(filename_save)):
        os.mkdir(os.path.dirname(filename_save))
    df.to_csv(filename_save, sep=';')

    return


if __name__ == '__main__':

    random.seed(123)
    seeds = random.randint(10000, size=(100,))

    folder_out = os.path.join(os.path.dirname(__file__), 'bert_eval_wordpiece')

    for seed in seeds[:]:
        main(seed,
             folder_out=folder_out
             )
