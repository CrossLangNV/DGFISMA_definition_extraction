from pathlib import Path

import plac

from data.load import SentenceData, LabelData
from models.preconfigured import BERTForDefinitionClassification


@plac.annotations(
    train_sentences=("Location of training sentences",),
    train_labels=("Location of training labels",),
    model_storage_directory=("Location of output model directory",),
    validation_split=("Fraction of training data to use as validation data", "option"),
    validation_sentences=("Location of validation sentences", "option"),
    validation_labels=("Location of validation labels", "option"),
    epochs=("number of epochs to train", "option")
)
def main(train_sentences: Path,
         train_labels: Path,
         model_storage_directory: Path,
         validation_split: float = .2,
         validation_sentences: Path = None,
         validation_labels: Path = None,
         epochs: int = 10,
         ):
    """Train the model

    Args:
        train_sentences: path to file that contains input sentences
            example file:
                "This is sentence 1\n
                While this is another sentence\n
                "
        train_labels: path to file that contains the labels (all integers)
            example file:
                "1\n
                0\n
                "
        model_storage_directory: path to where model should be saved
        validation_split: fraction of training data to be used as validation data.
            If validation_sentences and validation_labels are both not None,
            this will be ignored and whole training dataset is used.
        validation_sentences: path to file with validation dataset sentences. Same format as train_sentences.
        validation_labels: path to file with validation dataset labels. Same format as train_labels.
        epochs: number of epochs to train the model.

    Returns:
        trained model is saved in model_dir
    """

    train_sentences = Path(train_sentences)
    train_labels = Path(train_labels)
    model_storage_directory = Path(model_storage_directory)

    if validation_sentences is not None and validation_labels is not None:
        validation_sentences = Path(validation_sentences)
        validation_labels = Path(validation_labels)
    else:
        assert 0 <= validation_split < 1, f'Validation split should be in range [0, 1[. Got {validation_split}'

    # Load data
    train_sentences = SentenceData.from_file(train_sentences)
    train_labels = LabelData.from_file(train_labels)

    if validation_sentences is not None and validation_labels is not None:
        validation_sentences = SentenceData.from_file(validation_sentences)
        validation_labels = LabelData.from_file(validation_labels)

        validation_data = (validation_sentences, validation_labels)

    assert len(train_sentences) == len(train_labels), 'sentences and labels do not seem to match'

    # Initialise model and load pretrained weights

    model = BERTForDefinitionClassification.from_distilbert(model_storage_directory)

    # Train model on provided data

    if validation_sentences is not None and validation_labels is not None:
        model.train(train_sentences, train_labels, validation_data=validation_data, epochs=epochs)
    else:
        model.train(train_sentences, train_labels, validation_data=validation_split, epochs=epochs)

    return model.model_directory


if __name__ == "__main__":
    plac.call(main)
