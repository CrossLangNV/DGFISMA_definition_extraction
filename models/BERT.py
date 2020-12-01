from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics.classification import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from transformers import AdamW
from transformers import DistilBertConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_distilbert import DistilBertTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer

from bert_classifier.src.models import DistilBertSequenceClassifier


class BERTForSentenceClassification(object):
    """
    Classifier based on pretrained BERT model

    Args:
        model_directory (Path): directory where model folder is saved.
        architecture_config (PredefinedConfig):
        model (PreTrainedModel):
        tokenizer (PreTrainedTokenizer):
        bert_model (str): name of the model type
        seed (int): value to set seed to make training reproducible.
        device (str): choose whether to use cpu or gpu. From ("cpu", "cuda:0")
        epoch_init (int): internally epoch count is saved. Can be updated if training further.
        logger (Logger): logger object that writes CMD output to log file.
    """

    def __init__(self,
                 model_directory: Path,
                 architecture_config: PredefinedConfig,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 bert_model: str = None,
                 seed: int = 20200922,
                 device: str = 'cuda:0',
                 epoch_init: int = 0,
                 logger=None
                 ):

        self.model_directory = Path(model_directory)

        self.logger = logger

        self.architecture_config = architecture_config

        self.tokenizer = tokenizer

        self.seed = seed
        if self.logger:
            self.logger.info(f"seed = {self.seed}")

        self.bert_model = bert_model

        model.to(device)
        self.model = model

        self.device = device
        self.model.to(self.device)

        # save the amount of training epochs done.
        self.epoch = epoch_init

    @classmethod
    def from_bert_model(cls,
                        model_storage_directory: Path,
                        output_dim: int,
                        bert_model: (str,) = 'distilbert-base-uncased',
                        *args,
                        **kwargs
                        ) -> BERTForSentenceClassification:
        """When building a new model, it is created based on a bert model.


        Args:
            model_storage_directory: Directory where model will be created
            output_dim: number of output channels for classification.
            bert_model: name of preexisting bert model.
            *args: Optional args that will be given to constructor
            **kwargs:  Optional kwargs that will be given to constructor

        Returns:
            a :class:`BERTForSentenceClassification` instance
        """

        model_directory = get_model_directory(model_storage_directory)

        logger = init_logger(model_directory)

        # * Config *
        architecture_config = PredefinedConfig.from_model(bert_model)
        architecture_config.__setattr__('num_labels', output_dim)
        # save the config file

        # * MODEL *
        # Set-up the model
        if 'distil' in bert_model:
            model = DistilBertSequenceClassifier.from_pretrained(bert_model,
                                                                 config=architecture_config)
        else:
            raise ValueError(f"unknown model: {bert_model}")

        # * Tokenizer *
        # for preprocessing
        tokenizer = PredefinedTokenizer.from_model(bert_model)

        cls._save_classifier(model_directory,
                             tokenizer=tokenizer,
                             architecture_config=architecture_config,
                             )

        return cls(model_directory,
                   architecture_config,
                   model,
                   tokenizer,
                   bert_model=bert_model,
                   logger=logger,
                   *args,
                   **kwargs)

    @classmethod
    def from_dir(cls,
                 model_directory: Path,
                 bert_model: (str,) = 'distilbert-base-uncased',
                 device: str = 'cuda:0',
                 *args, **kwargs,
                 ):
        """Loads a :class:`BERTForSentenceClassification` from directory

        Args:
            model_directory: directory where all model components are saved
            bert_model: type of BERT model the model is based on.
            device: specify whether CPU or GPU is used.
            *args: optional args
            **kwargs: optional kwargs

        Returns:
            a :class:`BERTForSentenceClassification` instance
        """
        model_directory = Path(model_directory)

        logger = init_logger(model_directory)

        # * Config *
        if 'distil' in os.path.basename(bert_model):
            architecture_config_class = DistilBertConfig
        else:
            raise ValueError(f"unknown model: {bert_model}")

        architecture_config = architecture_config_class.from_json_file(
            os.path.join(model_directory, 'config.json'))

        # * Model *
        # Set-up the model from config
        if 'distil' in bert_model:
            model = DistilBertSequenceClassifier(architecture_config)
        else:
            raise ValueError(f"unknown model: {bert_model}")

        # Load weights
        model_path = sorted([filename for filename in os.listdir(model_directory) if filename.endswith(".pth")])[-1]
        model_filename = os.path.join(model_directory, model_path)
        checkpoint = torch.load(model_filename, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])

        # Get the epoch from file.
        try:
            epoch_init = int(os.path.splitext(model_path)[0].rsplit("_", 1)[-1])
        except ValueError:
            # Unable to get epoch from filename.
            epoch_init = 1

        # * Tokenizer *
        # tokenizer
        if 'distil' in bert_model:
            vocab_file = os.path.join(model_directory, 'vocab.txt')

            with open(os.path.join(model_directory, 'tokenizer_config.json')) as f:
                config_tokenizer = json.load(f)
            tokenizer = DistilBertTokenizer(vocab_file=vocab_file, **config_tokenizer)
        else:
            raise ValueError(f"unknown model: {bert_model}")

        return cls(model_directory,
                   architecture_config,
                   model,
                   tokenizer,
                   bert_model=bert_model,
                   device=device,
                   epoch_init=epoch_init,
                   logger=logger,
                   *args, **kwargs)

    @classmethod
    def _save_classifier(cls,
                         model_directory,
                         tokenizer=None,
                         architecture_config: PredefinedConfig = None):
        """To save classifier components in model directory

        Args:
            model_directory: where tokenizer and config should be saved.
            tokenizer: optional
            architecture_config: optional
        """
        # save the tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(model_directory)

        if architecture_config is not None:
            architecture_config.save_pretrained(model_directory)

    def _prepocessing_x(self, sentences: List[str]):
        """Preprocessing on input sentences before giving it to the model.

        Args:
            sentences (List[str]): list with input sentences

        Returns:
            list with matching list of word indices and corresponding list with attention masks.
        """
        self.logger.info("tokenizing...")

        # add special tokens and keep length in bounds
        # adds [CLS] and appends [SEP]
        encoding = [self.tokenizer.encode_plus(sent,
                                               add_special_tokens=True,
                                               max_length=self.architecture_config.max_position_embeddings,
                                               pad_to_max_length=True,  # make sure every input is of same length
                                               return_attention_mask=True
                                               ) for sent in sentences]

        sentences_ids, attention_masks = zip(
            *[(encoding_i['input_ids'], encoding_i['attention_mask']) for encoding_i in encoding])

        return sentences_ids, attention_masks

    def train(self,
              x_train,
              y_train,
              validation_data=None,
              validation_fraction=.2,
              train_bert: str = 'last_layer',
              learning_rate: float = 2e-5,
              epochs: int = 10):
        """Train and update the model

        Args:
            x_train: list of input sentences
            y_train: list of labels
            validation_data: (x, y) with validation data
            validation_fraction: Is only used when validation data is not None
            train_bert: what layers of bert to train
                'last_layer': only trains last layer
                'all_layers': all layers are trained
            epochs: amount of epochs to train
            learning_rate: learning rate for optimizer

        Returns:
            None
        """

        assert train_bert in ['last_layer', 'all_layers'], f'Unknown value for train_bert" {train_bert}'

        x_id_train, x_mask_train = self._prepocessing_x(x_train)

        if validation_data is not None:
            x_valid, y_valid = validation_data
            x_id_valid, x_mask_valid = self._prepocessing_x(x_valid)
        else:
            x_id_train, x_id_valid, x_mask_train, x_mask_valid, y_train, y_valid = train_test_split(
                x_id_train,
                x_mask_train,
                y_train,
                random_state=self.seed)

        dataloader_train = DataLoaderBERT.from_training(x_id_train, x_mask_train, y_train)
        dataloader_valid = DataLoaderBERT.from_validation(x_id_valid, x_mask_valid, y_valid)

        if train_bert == 'last_layer':
            self.model.freeze_bert_encoder()
            self.model.unfreeze_bert_encoder_last_layers()
        elif train_bert == 'all_layers':
            self.model.unfreeze_bert_encoder()
        else:
            raise ValueError(f'Unknown value for train_bert" {train_bert}')

        # Training

        optimizer = self.get_optimizer(learning_rate)

        loss_tr_last = self.train_function(dataloader_train,
                                           optimizer,
                                           epochs=epochs,
                                           dataloader_valid=dataloader_valid,
                                           )

        # Save the model
        self._save_model(optimizer,
                         loss_tr_last,
                         learning_rate=learning_rate
                         )

    def _save_model(self,
                    optimizer,
                    loss_tr,
                    learning_rate):
        """Saves the model

        Args:
            optimizer: Used optimizer
            loss_tr: Training loss
            learning_rate: used learning rate of optimizer
        """
        filename_pth = os.path.join(self.model_directory, f"{self.bert_model}_model_{self.epoch}.pth")
        self.logger.info(f"saving model in {filename_pth}.pth")

        # save the model
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_tr,
            'learning_rate': learning_rate
        }, filename_pth)

    def predict(self,
                x: List[str], verbose=0):
        """The model predicts the labels for x

        Args:
            x (List[str]): list of sentences to be predicted.

        Returns:
            list of labels and list of certainties of each class.
        """

        sentences_ids, attention_masks = self._prepocessing_x(x)

        prediction_dataloader = DataLoaderBERT.from_prediction(sentences_ids, attention_masks)

        self.logger.info("start inference...")
        # Put model in evaluation mode
        self.model.eval()

        prediction_proba_lst = []

        n_x = len(x)
        n_temp = 0

        for batch in prediction_dataloader:
            # Add batch to GPU if selected
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, attention_mask=b_input_mask)[0]

            prediction_proba = F.softmax(logits, dim=1).detach().cpu().numpy()

            prediction_proba_lst.append(prediction_proba)

            if verbose:
                n_temp += len(prediction_proba)
                self.logger.info(f'\tInference: {n_temp / n_x:.1%}')

        pred = np.concatenate(prediction_proba_lst, axis=0)
        predictions_labels = np.argmax(pred, axis=1)

        return predictions_labels, pred

    def get_optimizer(self, learning_rate):
        """Initialises and returns an AdamW optimizer

        Args:
            learning_rate: to configure the optimer

        Returns:
            the AdamW optimizer
        """
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                          correct_bias=False)  # to reproduce BertAdam specific behavior set correct_bias=False

        return optimizer

    def train_function(self,
                       dataloader_train: DataLoader,
                       optimizer,
                       epochs: int = 1,
                       dataloader_valid: DataLoader = None
                       ):
        """Trains the Torch model given the preconfigured dataloader.

        Args:
            dataloader_train (DataLoader):
            optimizer: Torch optimizer
            epochs (int): amount of epochs to train
            dataloader_valid (DataLoader): Optional, dataloader for validation data to validate on.

        Returns:
            None
        """

        def unpack_gpu(batch):
            """Add batch to GPU and unpack the inputs from our dataloader

            Args:
                batch: a batch from dataloader.

            Returns:
                b_input_ids, b_input_mask, b_labels
            """
            #
            b_input_ids, b_input_mask, b_labels = (t.to(self.device) for t in batch)
            return b_input_ids, b_input_mask, b_labels

        max_grad_norm = 1.0
        train_loss_set = []

        # BERT training loop
        for epoch in trange(epochs, desc="Epoch"):

            # TRAINING

            # Set our model to training mode
            self.model.train()
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # Train the data for one epoch
            for step, batch in enumerate(dataloader_train):
                b_input_ids, b_input_mask, b_labels = unpack_gpu(batch)

                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, _ = self.model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss_value = loss.item()
                train_loss_set.append(loss_value)
                # Backward pass
                loss.backward()
                # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_grad_norm)
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                # Update tracking variables
                tr_loss += loss_value
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            self.logger.info(f"epoch: {epoch + 1}")
            self.logger.info("Train loss: {}".format(tr_loss / nb_tr_steps))

            # VALIDATION

            def get_f1(y_true, y_pred):
                """

                Args:
                    y_true:
                    y_pred:

                Returns:

                """
                report = classification_report(y_true, y_pred, output_dict=True)
                f1 = report['weighted avg']['f1-score']

                return f1

            callback = {'f1': get_f1}
            callback_scores = {key: [] for key in callback}

            if dataloader_valid is not None:
                # Put model in evaluation mode
                self.model.eval()
                # Tracking variables
                eval_loss, eval_accuracy = 0, 0
                nb_eval_steps, nb_eval_examples = 0, 0
                # Evaluate data for one epoch

                for batch in dataloader_valid:

                    b_input_ids, b_input_mask, b_labels = unpack_gpu(batch)

                    # Telling the model not to compute or store gradients, saving memory and speeding up validation
                    with torch.no_grad():
                        # Forward pass, calculate logit predictions
                        logits = self.model(input_ids=b_input_ids, attention_mask=b_input_mask)[0]
                        # Move logits and labels to CPU
                    logits = logits.detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()

                    pred = np.argmax(logits, axis=-1)

                    for key in callback:
                        score_i = callback[key](label_ids, pred)
                        callback_scores[key].append(score_i)

                    tmp_eval_accuracy = flat_accuracy(label_ids, logits)

                    eval_accuracy += tmp_eval_accuracy
                    nb_eval_steps += 1

                eval_accuracy = eval_accuracy / nb_eval_steps

                self.logger.info("Validation Accuracy: {}".format(eval_accuracy))

                for key in callback_scores:
                    self.logger.info(f"Validation {key}: {np.mean(callback_scores[key])}")

            self.logger.info("\n")

        self.epoch += epochs  # update number of epochs trained.
        return train_loss_set[-1]

    def get_model(self):
        """Returns the BERT model

        Returns:
            the BERT model
        """
        return self.model


def get_model_directory(model_storage_directory) -> Path:
    """Make a new directory based on current time in the provided directory.

    Args:
        model_storage_directory:

    Returns:
        Path to model directory
    """
    timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    model_directory = os.path.join(model_storage_directory, timestamp)  # directory
    os.makedirs(model_directory, exist_ok=True)

    return Path(model_directory)


def init_logger(model_directory):
    """Besides printing intermediate results to the command line, it is also saved in a log file.
    call logger.info("<String to write to log.txt>") instead of print.

    Args:
        model_directory: directory where the log.txt is saved.

    Returns:
        The logger object
    """
    log = logging.getLogger()
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)

    return log


class PredefinedConfig(PretrainedConfig):
    """
    Predefined configs
    """

    @staticmethod
    def from_model(bert_model):
        """

        :param bert_model: name of model
        :return:
        """

        if 'distil' in bert_model:
            architecture_config_class = DistilBertConfig
        else:
            raise ValueError(f'Unknown BERT model: {bert_model}')

        config = architecture_config_class.from_pretrained(bert_model)
        return config


class PredefinedTokenizer(PreTrainedTokenizer):
    """
    Predefined tokenizers
    """

    @staticmethod
    def from_model(bert_model):
        """

        :param bert_model: name of model
        :return:
        """

        if 'distil' in bert_model:
            # For when padding to max length, it's filled on right side with 0's'
            tokenizer = DistilBertTokenizer.from_pretrained(bert_model,
                                                            do_lower_case=True,
                                                            padding_side='right',
                                                            )
        else:
            raise ValueError(f'Unknown tokenizer for: {bert_model}')

        return tokenizer


class DataLoaderBERT(DataLoader):
    """
    Predefined data-loaders for training, validation and test datasets.
    """

    @classmethod
    def from_training(cls,
                      training_inputs,
                      training_masks,
                      training_labels,
                      batch_size: int = 32
                      ) -> DataLoader:
        """Uses a random sampler

        Args:
            training_inputs: list with word indices
            training_masks: list with masks
            training_labels: list of labels
            batch_size: amount of samples in each batch

        Returns:
            A :class:`DataLoader` object
        """

        training_data = TensorDataSetBERT.from_lists(training_inputs,
                                                     training_masks,
                                                     training_labels)

        training_sampler = RandomSampler(training_data)

        return cls(training_data,
                   sampler=training_sampler,
                   batch_size=batch_size)

    @classmethod
    def from_validation(cls,
                        validation_inputs,
                        validation_masks,
                        validation_labels,
                        batch_size: int = 32
                        ) -> DataLoader:
        """Uses a sequential sampler

        Args:
            validation_inputs: list with word indices
            validation_masks: list with masks
            validation_labels: list of labels
            batch_size: amount of samples in each batch

        Returns:
            A :class:`DataLoader` object
        """

        validation_data = TensorDataSetBERT.from_lists(validation_inputs, validation_masks, validation_labels)

        validation_sampler = SequentialSampler(validation_data)

        return cls(validation_data, sampler=validation_sampler, batch_size=batch_size)

    @classmethod
    def from_prediction(cls,
                        prediction_inputs,
                        prediction_masks,
                        batch_size: int = 32
                        ) -> DataLoader:
        """Uses a sequential sampler

        Args:
            prediction_inputs: list with word indices
            prediction_masks: list with masks
            batch_size: amount of samples in each batch

        Returns:
            A :class:`DataLoader` object
        """

        validation_data = TensorDataSetBERT.from_lists(prediction_inputs, prediction_masks)

        validation_sampler = SequentialSampler(validation_data)

        return cls(validation_data, sampler=validation_sampler, batch_size=batch_size)


class TensorDataSetBERT(TensorDataset):
    """
    Simplified :class:`TensorDataSet` that can automatically convert lists to tensors.
    """

    @classmethod
    def from_lists(cls, *lists: list) -> TensorDataset:
        """Automatically casts the lists to tensors before building the :class:`TensorDataSet` object.

        Args:
            *lists: 1 or more lists that have to be cast

        Returns:
            TensorDataset object
        """
        tensors = [torch.tensor(list_i) for list_i in lists]

        return cls(*tensors)


def flat_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """Calculates accuracy of multi-label predictions.

    Args:
        labels: Array of shape (n, ), contains the index (in [0, k-1]) of the correct label.
        preds: Array of shape (n, k) that contains a likelihood of each label.


    Returns:
        accuracy
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)  # np.sum(pred_flat == labels_flat) / len(labels_flat)
