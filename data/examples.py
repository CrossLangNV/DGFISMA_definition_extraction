from typing import List

from sklearn.model_selection import train_test_split

from data.load import get_gold_standard, get_training_data
from .preprocessing import tokenize


def data_main(seed,
              val_ratio):
    """ Data generator for training and evaluating different models.

    Args:
        seed:
        val_ratio:

    Returns:

    """
    x_test, y_test = get_gold_standard()

    x, y = get_training_data()

    x_test = preprocessing(tokenize(x_test))
    x = preprocessing(tokenize(x))

    x_train, x_valid, y_train, y_valid = train_test_split(x, y,
                                                          random_state=seed,
                                                          shuffle=True,
                                                          test_size=val_ratio)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def preprocessing(x: List[str]) -> List[str]:
    """ Lowercases the strings

    Args:
        x:

    Returns:

    """
    return [x_i.lower() for x_i in x]
