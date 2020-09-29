from pathlib import Path

import plac
from sklearn.metrics import classification_report

from data.load import read_lines
from models.BERT import init_logger


@plac.annotations(
    path_y=("Filename of labels",),
    path_pred=("Filename of predictions",),
    path_logger=("Filename of logger", "option"),
)
def main(path_y: Path,
         path_pred: Path,
         path_logger: Path = '.'):
    """

    :return:
    """

    path_y = Path(path_y)
    path_pred = Path(path_pred)

    logger = init_logger(path_logger)

    logger.info(f"   y: {path_y}")
    logger.info(f"pred: {path_pred}")

    labels = read_lines(path_y)
    pred = read_lines(path_pred)

    labels = list(map(int, labels))
    pred = [int(p.split(maxsplit=1)[0]) for p in pred]

    report = classification_report(labels, pred, output_dict=True)

    report_str = classification_report(labels, pred)
    logger.info(report_str)

    return report


if __name__ == "__main__":
    plac.call(main)
