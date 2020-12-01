import os
from typing import List

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../tests/test_files/arne')
DATA_FOLDER = os.path.normpath(DATA_FOLDER)
assert os.path.exists(DATA_FOLDER)


class DefinitionData(tuple):
    """
    Tuple for definition data that links sentence and label list
    """

    def __new__(cls, sentences: List[str], labels: List[int]):
        """Construct (sentences, labels) tuple

        Args:
            sentences: list of definition sentences
            labels: list of classification indices
        """
        sentences = SentenceData(sentences)

        assert isinstance(labels, list)

        assert len(sentences) == len(labels), 'x and y should be of equal length'

        for s in labels:
            assert isinstance(s, int)

        return super(DefinitionData, cls).__new__(cls, (sentences,
                                                        labels))


class SentenceData(list):
    """Class for sentence data

    Args:
        sentences: list of sentences
    """

    def __init__(self, sentences: List[str]):
        super(SentenceData, self).__init__(sentences)

        for s in self:
            assert isinstance(s, str)

    @classmethod
    def from_file(cls, path=os.path.join(DATA_FOLDER, 'test_sentences')):
        """ Return :class:`SentenceData` object from a file

        Args:
            path: path to file with sentences

        Returns:
            :class:`SentenceData` object
        """

        return cls(read_lines(path))


class LabelData(list):
    """Class for label data

    Args:
        labels: list of labels
    """

    def __init__(self, labels: List[int]):
        super(LabelData, self).__init__(labels)

        for s in self:
            assert isinstance(s, int)

    @classmethod
    def from_file(cls, path=os.path.join(DATA_FOLDER, 'test_labels')):
        """ Return :class:`LabelData` object from a file

        Args:
            path: path to file with indices

        Returns:
            :class:`LabelData` object
        """

        return cls(map(int, read_lines(path)))


def get_gold_standard() -> DefinitionData:
    """
    Predefined gold standard made by Arne
    """

    sentences = SentenceData.from_file()

    labels = LabelData.from_file()

    return DefinitionData(sentences, labels)


def get_training_data(path_train=None,
                      delimiter=None):
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


def read_lines(path):
    """
    Reads and lists text lines of document
    """

    with open(path, 'r') as f:
        return [s.strip() for s in f.read().strip("\n").split("\n")]
