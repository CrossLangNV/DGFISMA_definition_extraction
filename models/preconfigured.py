from pathlib import Path
from typing import List

from models.BERT import BERTForSentenceClassification


class BERTForDefinitionClassification(BERTForSentenceClassification):
    """Class inherited from :class:`BERTForSentenceClassification` to more easily build models for
    definition extraction
    """

    @classmethod
    def from_distilbert(cls, model_storage_directory: Path) -> BERTForSentenceClassification:
        """Preconfigured DistilBERT model for 2 class sentence classification

        Args:
            model_storage_directory (Path): directory where new model folder
                will be generated.

        Returns:
            a :class:`BERTForSentenceClassification` model, pretrained on BERT.
        """

        return cls.from_bert_model(model_storage_directory=model_storage_directory,
                                   output_dim=2,
                                   bert_model='distilbert-base-uncased'
                                   )

    def train(self,
              x_train: List[str],
              y_train: List[int],
              validation_data: (float, tuple),
              *args,
              **kwargs):
        """Train the model with validation data that can be either a fraction to
        be extracted from the training data, or a predefined validation set.

        Args:
            x_train (List[str]): list with sentences to be classified
            y_train (List[int]): list with ground truth index per sentence.
            validation_data (float, tuple): can be either a fraction or a tuple
                with (x_valid, y_valid)
            *args: optional args
            **kwargs: optional kwargs

        Returns:
            None
        """

        # decide if validation data is a fraction or predefined validation set.
        if isinstance(validation_data, float):
            super(BERTForDefinitionClassification, self).train(x_train, y_train,
                                                               validation_fraction=validation_data, *args, **kwargs)
        else:
            super(BERTForDefinitionClassification, self).train(x_train, y_train,
                                                               validation_data=validation_data, *args, **kwargs)
