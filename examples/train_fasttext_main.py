"""
* Load data
* Split data
* Init a model
* Train
* Evaluate
"""
import os
import tempfile
from typing import List

import fasttext
import numpy as np
import pandas as pd
from numpy import random
from sklearn.metrics import classification_report

from data.examples import data_main


class ModelFasttext(object):
    labels = ['__label__no_definition', '__label__definition']

    def __init__(self,
                 seed,
                 ):
        self.seed = seed

    def get_model(self):
        return self.model

    def train(self, x_train, y_train, validation: List[list]):
        """
        Train the fasttext model by autotuning on validation set.
        """

        with tempfile.NamedTemporaryFile() as f_valid:
            filename_valid = f_valid.name

            with open(filename_valid, "w") as f:
                for text, label in zip(*validation):
                    f.write(f'{self.labels[label]} {text}\n')

            with tempfile.NamedTemporaryFile() as f:
                filename_train = f.name

                with open(filename_train, "w") as f:
                    for text, label in zip(x_train, y_train):
                        f.write(f'{self.labels[label]} {text}\n')

                """
                The following can be optimized:
                Warning : bucket is manually set to a specific value. It will not be automatically optimized.
                Warning : wordNgrams is manually set to a specific value. It will not be automatically optimized.
                Warning : dim is manually set to a specific value. It will not be automatically optimized.
                Warning : lr is manually set to a specific value. It will not be automatically optimized.
                Warning : epoch is manually set to a specific value. It will not be automatically optimized.
                """

                model = fasttext.train_supervised(
                    input=filename_train,
                    minCount=5,
                    ws=1,
                    seed=self.seed,
                    thread=1,  # To make fully fully reproducible https://fasttext.cc/docs/en/faqs.html
                    autotuneValidationFile=filename_valid,
                )

        self.model = model

    def predict(self, x):
        def get_index(pred):
            return [int(label[0] == '__label__definition') for label in pred]

        pred = self.get_model().predict(list(x))[0]
        pred = get_index(pred)

        return pred

    def get_model_info(self):
        return {'dim': self.get_model().get_dimension(),
                'epochs': self.get_model().epoch,
                'wordNgrams': self.get_model().wordNgrams,
                'lr': self.get_model().lr,
                'bucket': self.get_model().bucket}


def main(seed,
         folder_out='.',
         val_ratio=.2,
         ):
    """
    Produce a csv with performance of training fasttext for definition extraction
    """

    # Data preparation
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_main(seed,
                                                                         val_ratio=val_ratio)

    # Preparing model
    model_fasttext = ModelFasttext(seed)

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

    delta = .1
    n_train_tot = len(x_train)
    # for split in np.arange(1, 0, -delta)[::-1]:
    for split in [1]:
        n_split = int(np.round(split * n_train_tot))

        x_train_split = x_train[:n_split]
        y_train_split = y_train[:n_split]

        model_fasttext.train(x_train_split, y_train_split, validation=(x_valid, y_valid))

        model_info = model_fasttext.get_model_info()

        pred_train_split = model_fasttext.predict(x_train_split)
        pred_valid = model_fasttext.predict(x_valid)
        pred_test = model_fasttext.predict(x_test)

        n_train = len(x_train_split)
        for dataset, x, y, pred in [
            ('train', x_train_split, y_train_split, pred_train_split),
            ('validation', x_valid, y_valid, pred_valid),
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

    for seed in seeds:
        main(seed=seed,
             folder_out=folder_out
             )
