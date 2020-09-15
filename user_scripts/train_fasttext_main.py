"""
* Load data
* Split data
* Init a model
* Train
* Evaluate
"""
import tempfile

import fasttext
import numpy as np
from numpy import random
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


from data.load import get_training_data, get_gold_standard


def data_main(seed,
              val_size):
    # TODO refactor to be used by both trainers (Fasttext and BERT)

    x_test, y_test = get_gold_standard()

    x, y = get_training_data()

    x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                          random_state=seed,
                                                          shuffle=True,
                                                          test_size=val_size)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


class ModelFasttext(object):
    labels = ['__label__no_definition', '__label__definition']

    def __init__(self, seed):
        self.seed = seed

    def get_model(self):
        return self.model

    def train(self, x_train, y_train):

        with tempfile.NamedTemporaryFile() as f:
            filename_train = f.name

            with open(filename_train, "w") as f:
                for text, label in zip(x_train, y_train):
                    f.write(f'{self.labels[label]} {text}\n')

            model = fasttext.train_supervised(
                input=filename_train,
                epoch=200,
                dim=100,
                minCount=5,
                ws=1,
                seed=self.seed,
                thread=1,   # To make fully fully reproducible https://fasttext.cc/docs/en/faqs.html
            )
        self.model = model

    def predict(self, x):
        def get_index(pred):
            return [int(label[0] == '__label__definition') for label in pred]

        pred = self.get_model().predict(list(x))[0]
        pred = get_index(pred)

        return pred


def main(seed=15092020,
         val_size=.2,
         b_save=False):
    # Data preparation
    (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = data_main(seed,
                                                                         val_size=val_size)

    # Preparing model
    model_fasttext = ModelFasttext(seed)

    def get_info(n_train,
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
    for split in np.arange(1, 0, -delta)[::-1]:
        n_split = int(np.round(split * n_train_tot))

        x_train_split = x_train[:n_split]
        y_train_split = y_train[:n_split]

        model_fasttext.train(x_train_split, y_train_split)

        pred_train_split = model_fasttext.predict(x_train_split)
        pred_valid = model_fasttext.predict(x_valid)
        pred_test = model_fasttext.predict(x_test)

        n_train = len(x_train_split)
        for dataset, x, y, pred in (
                ('train', x_train_split, y_train_split, pred_train_split),
                ('validation', x_valid, y_valid, pred_valid),
                ('test', x_test, y_test, pred_test)
        ):

            report = classification_report(y, pred, output_dict=True)

            for label in ['0', '1', 'weighted avg']:
                info = get_info(n_train,
                                report,
                                label,
                                dataset)
                l.append(info)

    df = pd.DataFrame(l)

    df.to_csv(f'fasttext_eval/eval_s{seed}.csv', sep=';')

    return


if __name__ == '__main__':

    random.seed(123)
    seeds = random.randint(10000, size=(100,))

    for seed in seeds:
        main(b_save=False, seed=seed)
