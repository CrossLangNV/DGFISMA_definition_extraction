import argparse
import logging
import os
import socket
import time
from pathlib import Path

import numpy as np
import plac
import torch
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from transformers import BertConfig, AdamW
from transformers import DistilBertConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer

from bert_classifier.src.models import BertSequenceClassifier, DistilBertSequenceClassifier, \
    DistilBertGRUSequenceClassifier
from bert_classifier.src.utils import csv_field_limit
from data.load import get_training_data

csv_field_limit()


def get_parser(argv=None):
    parser = argparse.ArgumentParser()
    # Input-output:
    parser.add_argument("--filename", dest="FILENAME",
                        help='Path to the training data. A csv with at each line: "plain text"￭label_tag￭label_nr. The plain text should be enclosed in ".."',
                        required=True)
    parser.add_argument("--delimiter", dest="DELIMITER",
                        help="delimiter used in input csv", required=False, type=str, default="￭")
    parser.add_argument("--model_storage_directory", dest="MODEL_STORAGE_DIRECTORY",
                        help="output directory", required=True)
    # Preprocessing parameters:
    parser.add_argument("--train_size", dest="TRAIN_SIZE",
                        help="Fraction of data used for training. 1-TRAIN_SIZE will be used for validation",
                        required=False, type=float, default=0.8)
    parser.add_argument("--seed", dest="SEED",
                        help="seed used for train/val split", required=False, type=int, default=2018)
    # neural network parameters:
    parser.add_argument("--output_dim", dest="OUTPUT_DIM",
                        help="Nr of output classes. Defaults to the binary case", required=False, type=int, default=2)
    parser.add_argument('--bert_model', dest="BERT_MODEL",
                        help='bert model chosen: bert or distilbert architecture, e.g. "bert-base-uncased", "distilbert-base-uncased",...',
                        default='distilbert-base-uncased')
    parser.add_argument('--train_bert', dest="TRAIN_BERT",
                        help='whether to train all layers of BERT, freeze all layers of BERT, or only train last layer of BERT',
                        choices=['freeze_all_layers', 'last_layer', 'all_layers'], default='all_layers')
    parser.add_argument('--rnn_model', dest='RNN_MODEL',
                        help='send hidden states of BERT to RNN instead of using the BERT pooling layers. Only available in combination with distilbert',
                        action='store_true')
    parser.add_argument('--rnn_bidirectional', dest='RNN_BIDIRECTIONAL',
                        help='Bidirectional rnn layer. Only used when --rnn_model is used, otherwise ignored.',
                        action='store_true')
    parser.add_argument("--rnn_hidden_dim", dest="RNN_HIDDEN_DIM",
                        help="Hidden dimension of the RNN. Only used when --rnn_model is used, otherwise ignored.",
                        required=False, type=int, default=256)
    parser.add_argument("--rnn_n_layers", dest="RNN_N_LAYERS",
                        help="Number of RNN layers. Only used when --rnn_model is used, otherwise ignored.",
                        required=False, type=int, default=2)
    parser.add_argument("--rnn_dropout", dest="RNN_DROPOUT",
                        help="Dropout on the rnn layer if n_layers>2. Only used when --rnn_model is used, otherwise ignored.",
                        required=False, type=float, default=0.25)
    parser.add_argument("--batch_size", dest="BATCH_SIZE",
                        help="batch size", required=False, type=int, default=32)
    parser.add_argument("--epochs", dest="EPOCHS",
                        help="nr of training epochs", required=False, type=int, default=10)
    parser.add_argument("--learning_rate", dest="LEARNING_RATE",
                        help="initial learning rate of adam optimizer", required=False, type=float, default=2e-5)
    # device:
    parser.add_argument("--device", dest="DEVICE",
                        help="device: cuda:0, cuda:1,...cpu", required=False, type=str, default="cpu")

    args = parser.parse_args(argv)

    return args


def main_arg(filename: str,
             model_storage_directory: str,
             delimiter: str = None,
             epochs: int = None,
             batch_size: int = None
             ):
    argv = ['--filename', filename,
            '--model_storage_directory', model_storage_directory,
            ]

    if delimiter is not None:
        argv += ['--delimiter', delimiter]

    if epochs is not None:
        argv += ['--epochs', str(epochs)]

    if batch_size is not None:
        argv += ['--batch_size', str(batch_size)]

    main(argv)


@plac.annotations(
    filename=plac.Annotation(
        'Path to the training data. A csv with at each line: "plain text"￭label_tag￭label_nr. The plain text should be enclosed in ".."',
        'positional', type=Path),
    delimiter=plac.Annotation('Delimiter used in input csv',
                              'option', abbrev='del', type=str),
    model_storage_directory=plac.Annotation("output directory",
                                            "positional", type=Path),
    # device=()
    # train_size=()
    # val_size=()
    # seed=()
    # output_dim=()
    # bert_model=()
    # train_bert=(),
    # rnn_model=(),
    # rnn_bidirectional=(),
    # rnn_hidden_dim=(),
    # rnn_n_layers=(),
    # rnn_dropout=(),
    batch_size=plac.Annotation("batch size", 'option', abbrev='b', type=int),
    epochs=plac.Annotation("nr of training epochs", 'option', abbrev='e', type=int),
    # learning_rate=(),
)
def main(
        # Input-output:
        filename: Path,
        model_storage_directory: Path,
        delimiter: str = "￭",  # delimiter of the input csv
        # Preprocessing parameters:
        train_size: float = .8,
        validation_filename: Path = None,
        seed: int = 2018,
        output_dim: int = 2,
        bert_model: str = 'distilbert-base-uncased',
        train_bert: str = 'last_layer',
        # RNN:
        rnn_model: bool = None,
        rnn_bidirectional=None,
        rnn_hidden_dim: int = 256,
        rnn_n_layers: int = 2,
        rnn_dropout: float = .25,
        # Fitting hyper parameters
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 2e-5,
        device='cpu',  # "cuda:0", cpu,..
):
    # 2) Set run specific envirorment configurations

    timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    model_directory = os.path.join(model_storage_directory, timestamp)  # directory
    os.makedirs(model_directory, exist_ok=True)

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

    # 3) Set up config:

    bert_model_path = bert_model

    if 'distil' in bert_model_path:
        ArchitectureConfig = DistilBertConfig
    else:
        ArchitectureConfig = BertConfig

    if os.path.exists(bert_model_path):
        if os.path.exists(os.path.join(bert_model_path, CONFIG_NAME)):
            print(f"loading {bert_model_path}")
            config = ArchitectureConfig.from_json_file(os.path.join(bert_model_path, CONFIG_NAME))
        elif os.path.exists(os.path.join(bert_model_path, 'bert_config.json')):
            print(f"loading {bert_model_path}")
            config = ArchitectureConfig.from_json_file(os.path.join(bert_model_path, 'bert_config.json'))
        else:
            raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
    else:
        config = ArchitectureConfig.from_pretrained(bert_model_path)

    config.__setattr__('num_labels', output_dim)

    if rnn_model:
        # only necessary for GRUDistilBertClassifier
        config.__setattr__('rnn_bidirectional', rnn_bidirectional)
        config.__setattr__('rnn_hidden_dim', rnn_hidden_dim)
        config.__setattr__('rnn_n_layers', rnn_n_layers)
        config.__setattr__('rnn_dropout', rnn_dropout)

    # 3) Load the data:
    sentences, labels = get_training_data(filename, delimiter)

    # 4) Preprocess the data:
    if 'distil' in bert_model_path:
        tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)

    def sentences_prep(sentences):

        log.info("tokenizing...")
        tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

        # add special tokens
        tokenized_texts = [['[CLS]'] + sentence[:  (config.max_position_embeddings - 2)] + ["[SEP]"] for sentence in
                           tokenized_texts]

        # convert to ids+padding
        input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=config.max_position_embeddings, dtype="long", truncating="post",
                                  padding="post")

        # Create attention masks
        attention_masks = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        return input_ids, attention_masks

    input_ids, attention_masks = sentences_prep(sentences)

    if validation_filename is not None:

        sentences_valid, labels_valid = get_training_data(validation_filename, delimiter)
        input_ids_valid, attention_masks_valid = sentences_prep(sentences_valid)

        # Use train_test_split to split our data into train and validation sets for training
        train_inputs, validation_inputs, train_labels, validation_labels = input_ids, input_ids_valid, labels, labels_valid

        train_masks, validation_masks = attention_masks, attention_masks_valid

    else:
        val_size = 1.0 - train_size

        # Use train_test_split to split our data into train and validation sets for training
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                            random_state=seed,
                                                                                            test_size=val_size)
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                               random_state=seed, test_size=val_size)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Create an iterator of our data with torch DataLoader
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # 4) set-up the model:

    if 'distil' in bert_model_path:
        if rnn_model:
            model = DistilBertGRUSequenceClassifier.from_pretrained(bert_model_path, config=config)
        else:
            model = DistilBertSequenceClassifier.from_pretrained(bert_model_path, config=config)

    else:
        model = BertSequenceClassifier.from_pretrained(bert_model_path, config=config)

    model.to(device)

    if train_bert == 'freeze_all_layers':
        model.freeze_bert_encoder()
    elif train_bert == 'last_layer':
        model.freeze_bert_encoder()
        model.unfreeze_bert_encoder_last_layers()
    elif train_bert == 'all_layers':
        pass

    # 5) train the model:

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, pred_flat)  # np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train_func(epochs, model, optimizer, train_dataloader, validation_dataloader, device):

        max_grad_norm = 1.0
        train_loss_set = []

        # BERT training loop
        for epoch in trange(epochs, desc="Epoch"):

            ## TRAINING

            # Set our model to training mode
            model.train()
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            # Train the data for one epoch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                loss, _ = model(input_ids=b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                train_loss_set.append(loss.item())
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
                # Update parameters and take a step using the computed gradient
                optimizer.step()
                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            ## VALIDATION

            # Put model in evaluation mode
            model.eval()
            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            # Evaluate data for one epoch
            for batch in validation_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    logits = model(input_ids=b_input_ids, attention_mask=b_input_mask)
                    # Move logits and labels to CPU
                logits = logits[0].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            log.info(f"epoch: {epoch + 1}")
            log.info("Train loss: {}".format(tr_loss / nb_tr_steps))
            log.info("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
            log.info("\n")

        return tr_loss

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                      correct_bias=False)  # to reproduce BertAdam specific behavior set correct_bias=False

    tr_loss = train_func(epochs, model, optimizer, train_dataloader, validation_dataloader, device)

    # 7)Save the model:
    log.info(f"saving model in {model_directory}/{bert_model}_model_{epochs}.pth")
    # save the config file
    config.save_pretrained(model_directory)
    # save the tokenizer
    tokenizer.save_pretrained(model_directory)
    # save the model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': tr_loss,
        'learning_rate': learning_rate
    }, os.path.join(model_directory, f"{bert_model}_model_{epochs}.pth"))

    return model


if __name__ == "__main__":
    plac.call(main)
