import argparse
import os
from base64 import b64decode
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertConfig
from transformers import DistilBertConfig
from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer

from data.load import get_sentences
from .models import BertSequenceClassifier, DistilBertSequenceClassifier, DistilBertGRUSequenceClassifier


class DefExtractModel(object):
    def __init__(self,
                 model_path: str,
                 rnn_model: bool = False,
                 batch_size: int = 32,
                 device: str = "cpu"):

        self.rnn_model = rnn_model
        self.batch_size = batch_size
        self.device = device

        self.load_model(model_path)
        self.tokenizer = get_tokenizer(model_path=model_path)

    def load_model(self, model_path):

        # 2) Load the config, model and tokenizer
        # config:
        if 'distil' in os.path.basename(model_path):
            ArchitectureConfig = DistilBertConfig
        else:
            ArchitectureConfig = BertConfig

        model_dir = os.path.dirname(model_path)

        if os.path.exists(model_path):
            if os.path.exists(os.path.join(model_dir, 'config.json')):
                print(f"loading {model_path}")
                self.config = ArchitectureConfig.from_json_file(os.path.join(model_dir, 'config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            raise ValueError(f"Cannot find BERT based model at {model_path} you are attempting to load.")

        # model:
        if 'distil' in os.path.basename(model_path):
            if self.rnn_model:
                model = DistilBertGRUSequenceClassifier(self.config)
            else:
                model = DistilBertSequenceClassifier(self.config)
        else:
            model = BertSequenceClassifier(self.config)

        self.model = model

        checkpoint = torch.load(model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)

    def predict(self, sentences: List[str]):
        # Preprocess the data:
        prediction_dataloader = self.preprocessing(sentences)

        print("start inference...")
        # Put model in evaluation mode
        self.model.eval()
        # Tracking variables
        predictions_proba = []
        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = self.model(b_input_ids, attention_mask=b_input_mask)
            # Move logits and labels to CPU
            logits = logits[0]  # .detach().cpu().numpy()
            # Store predictions and true labels
            prediction_proba = F.softmax(logits, dim=1).detach().cpu().numpy()
            predictions_proba.append(prediction_proba)

        flat_predictions_proba = [item for sublist in predictions_proba for item in sublist]
        predictions_labels = np.argmax(flat_predictions_proba, axis=1).flatten()

        return predictions_labels, flat_predictions_proba

    def preprocessing(self, sentences):
        print("tokenizing...")
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in sentences]

        # add special tokens
        tokenized_texts = [['[CLS]'] + sentence[: (self.config.max_position_embeddings - 2)] + ["[SEP]"] for sentence in
                           tokenized_texts]

        # convert to ids+padding
        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                                  maxlen=self.config.max_position_embeddings, dtype="long", truncating="post",
                                  padding="post")

        # Create attention masks
        attention_masks = []
        # Create a mask of 1s for each token followed by 0s for padding
        for seq in input_ids:
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)

        # create test tensors
        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
        prediction_data = TensorDataset(prediction_inputs, prediction_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)

        return prediction_dataloader


def get_tokenizer(model_path):
    model_dir = os.path.dirname(model_path)

    # tokenizer
    if 'distil' in os.path.basename(model_path):
        tokenizer = DistilBertTokenizer.from_pretrained(model_dir, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=True)

    return tokenizer


def load_sentences(filepath, base_64=False):
    sentences = get_sentences(filepath)

    if base_64:
        sentences = [b64decode(sentence).decode() for sentence in sentences]

    return sentences


def main(args=None):
    parser = argparse.ArgumentParser()
    # Input-output:
    parser.add_argument("--filename", dest="FILENAME",
                        help='path to the test data (file with at each line a base64 encoded document if base_64 flag set to true)',
                        required=True)
    parser.add_argument('--base_64', dest='BASE_64', action='store_true')
    parser.add_argument("--model_path", dest="MODEL_PATH",
                        help="path to the trained nodel", required=True)
    parser.add_argument("--output_file", dest="OUTPUT_FILE",
                        help="output file with predicted labels", required=True)
    # neural network parameters:
    parser.add_argument('--rnn_model', dest='RNN_MODEL',
                        help='whether the pretrained model sends hidden states of BERT to a RNN instead of using the BERT pooling layers. Only available in combination with distilbert',
                        action='store_true')
    parser.add_argument("--batch_size", dest="BATCH_SIZE",
                        help="batch size used during inference", required=False, type=int, default=32)
    # device:
    parser.add_argument("--device", dest="DEVICE",
                        help="device: cuda:0, cuda:1,...cpu", required=False, type=str, default="cpu")
    args = parser.parse_args(args)

    # 1) Parameters:
    FILENAME = args.FILENAME
    BASE_64 = args.BASE_64
    MODEL_PATH = args.MODEL_PATH
    OUTPUT_FILE = args.OUTPUT_FILE
    DEVICE = args.DEVICE  # "cuda:0", cpu,..
    RNN_MODEL = args.RNN_MODEL
    BATCH_SIZE = args.BATCH_SIZE

    # 2) Load the data
    sentences = load_sentences(FILENAME, base_64=BASE_64)

    # Load model
    def_extract_model = DefExtractModel(MODEL_PATH,
                                        rnn_model=RNN_MODEL,
                                        batch_size=BATCH_SIZE,
                                        device=DEVICE)

    # Prediction
    predictions_labels, flat_predictions_proba = def_extract_model.predict(sentences)

    if OUTPUT_FILE is not None:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

        with open(OUTPUT_FILE, "w") as fp:
            for pred_label, pred_proba_label in zip(predictions_labels, flat_predictions_proba):
                fp.write(f"{pred_label} {pred_proba_label.tolist()}\n")

    return predictions_labels, flat_predictions_proba


if __name__ == "__main__":
    main()
