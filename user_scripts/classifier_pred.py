from pathlib import Path

import plac

from data.load import read_lines
from models.preconfigured import BERTForDefinitionClassification


@plac.annotations(
    model_dir=("Directory containing model",),
    path_x=("Filename of sentences",),
    path_pred=("Filename of where to write predictions", "option"),
)
def main(model_dir: Path,
         path_x: Path,
         path_pred: Path = None):
    """

    :return:
    """

    model_dir = Path(model_dir)
    path_x = Path(path_x)
    if path_pred is not None:
        path_pred = Path(path_pred)

    if path_pred is not None:
        """
        Save output to file
        """

    # Load data
    sentences = read_lines(path_x)

    model = BERTForDefinitionClassification.from_dir(model_dir)

    pred_labels, pred_proba = model.predict(sentences)

    if path_pred is not None:
        with open(path_pred, 'w+') as f:
            for label_i, proba_i in zip(pred_labels, pred_proba):
                f.write(f'{label_i} {proba_i}\n')

    return pred_labels, pred_proba


if __name__ == "__main__":
    plac.call(main)
