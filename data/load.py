import os

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../tests/test_files/arne')
DATA_FOLDER = os.path.normpath(DATA_FOLDER)
assert os.path.exists(DATA_FOLDER)


class DefinitionData(tuple):

    def __new__(cls, sentences, labels):
        sentences = SentenceData(sentences)

        assert isinstance(labels, list)

        assert len(sentences) == len(labels), 'x and y should be of equal length'

        for s in labels:
            assert isinstance(s, int)

        return super(DefinitionData, cls).__new__(cls, (sentences,
                                                        labels))


class SentenceData(list):
    def __init__(self, sentences):

        super(SentenceData, self).__init__(sentences)

        for s in self:
            assert isinstance(s, str)


def get_gold_standard() -> DefinitionData:
    """
    Predefined gold standard made by Arne
    """

    path_sentences = os.path.join(DATA_FOLDER, 'test_sentences')
    path_labels = os.path.join(DATA_FOLDER, 'test_labels')

    assert os.path.exists(path_sentences), f"can't find path sentences @ {path_sentences}"
    assert os.path.exists(path_labels), f"can't find path sentences @ {path_labels}"

    sentences = read_lines(path_sentences)
    labels = list(map(int, read_lines(path_labels)))

    return DefinitionData(sentences, labels)


def get_training_data(path_train=None, delimiter=None):
    """
    If path is None, default file is used
    """
    import pandas as pd
    import csv

    if path_train is None:
        path_train = os.path.join(DATA_FOLDER, 'train_dgfisma_wiki.csv')
        delimiter = "⚫"

    assert os.path.exists(path_train), f"can't find path sentences @ {path_train}"

    dataset = pd.read_csv(path_train, delimiter=delimiter, quoting=csv.QUOTE_NONE, header=None, engine='python',
                          names=['text', 'label_tag', 'label'])

    labels = dataset.label.tolist()
    sentences = dataset.text.tolist()
    sentences = [sentence.strip("\"\'") for sentence in sentences]  # sentences are enclosed in " "

    return DefinitionData(sentences, labels)


def get_sentences(path):
    return SentenceData(read_lines(path))


def read_lines(path):
    """
    Reads and lists textlines of document
    """
    with open(path, 'r') as f:
        return [s.strip() for s in f.read().strip("\n").split("\n")]
