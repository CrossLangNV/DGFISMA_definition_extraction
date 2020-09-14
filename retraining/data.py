import os

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '../tests/test_files/arne')
DATA_FOLDER = os.path.normpath(DATA_FOLDER)
assert os.path.exists(DATA_FOLDER)


def get_gold_standard():
    """
    Predefined gold standard made by Arne
    """

    path_sentences = os.path.join(DATA_FOLDER, 'test_sentences')
    path_labels = os.path.join(DATA_FOLDER, 'test_labels')

    assert os.path.exists(path_sentences), f"can't find path sentences @ {path_sentences}"
    assert os.path.exists(path_labels), f"can't find path sentences @ {path_labels}"

    sentences = read_lines(path_sentences)
    labels = list(map(int, read_lines(path_labels)))

    return sentences, labels


def read_lines(path):
    """
    Reads and lists textlines of document
    """
    with open(path, 'r') as f:
        return [s.strip() for s in f.read().strip("\n").split("\n")]
