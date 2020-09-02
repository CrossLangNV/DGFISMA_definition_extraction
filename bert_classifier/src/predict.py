import logging
import os
import time
import socket
import random
import pickle
import argparse
import csv
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm, trange

from base64 import b64encode, b64decode

import torch
import torch.nn.functional as F

from transformers import BertPreTrainedModel, BertConfig, BertModel, AdamW

from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer

from transformers import DistilBertConfig, DistilBertModel
from transformers.modeling_distilbert import DistilBertPreTrainedModel

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from models import BertSequenceClassifier, DistilBertSequenceClassifier, DistilBertGRUSequenceClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Input-output:
    parser.add_argument("--filename", dest="FILENAME",
                        help='path to the test data (file with at each line a base64 encoded document if base_64 flag set to true)', required=True)
    parser.add_argument('--base_64', dest='BASE_64', action='store_true')
    parser.add_argument("--model_path", dest="MODEL_PATH",
                        help="path to the trained nodel", required=True)
    parser.add_argument("--output_file", dest="OUTPUT_FILE",
                        help="output file with predicted labels", required=True)
    #neural network parameters:
    parser.add_argument('--rnn_model', dest='RNN_MODEL', help='whether the pretrained model sends hidden states of BERT to a RNN instead of using the BERT pooling layers. Only available in combination with distilbert' , action='store_true')
    parser.add_argument("--batch_size", dest="BATCH_SIZE",
                        help="batch size used during inference", required=False, type=int, default=32)
    #device:
    parser.add_argument("--device", dest="DEVICE",
                        help="device: cuda:0, cuda:1,...cpu", required=False, type=str, default="cpu")
    args = parser.parse_args()

    #1) Parameters:
    FILENAME=args.FILENAME
    BASE_64=args.BASE_64
    MODEL_PATH=args.MODEL_PATH
    OUTPUT_FILE=args.OUTPUT_FILE
    DEVICE =args.DEVICE  #"cuda:0", cpu,..
    RNN_MODEL=args.RNN_MODEL
    BATCH_SIZE=args.BATCH_SIZE
    
    MODEL_DIR=os.path.dirname(  MODEL_PATH )

    #2) Load the data
    
    #load the sentences
    sentences=open( FILENAME ).read().rstrip("\n").split("\n")
    if BASE_64:
        sentences=[ b64decode( sentence ).decode()  for sentence in sentences  ]
        
    #2) Load the config, model and tokenizer
    #config:
    if 'distil' in os.path.basename(MODEL_PATH):
        ArchitectureConfig=DistilBertConfig
    else:
        ArchitectureConfig=BertConfig

    if os.path.exists( MODEL_PATH  ):
        if os.path.exists(os.path.join( MODEL_DIR , 'config.json' )):
            print( f"loading {MODEL_PATH}" )
            config = ArchitectureConfig.from_json_file(os.path.join(MODEL_DIR, 'config.json' ))
        else:
            raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
    else: 
        raise ValueError( f"Cannot find BERT based model at {MODEL_PATH} you are attempting to load." )

    #model:
        
    if 'distil' in os.path.basename(MODEL_PATH):
        if RNN_MODEL:
            model = DistilBertGRUSequenceClassifier( config )
        else:    
            model = DistilBertSequenceClassifier( config )
    else:
        model = BertSequenceClassifier( config  )
    
    checkpoint = torch.load( MODEL_PATH , map_location=torch.device( DEVICE )  )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
        
    #tokenizer
    if 'distil' in os.path.basename(MODEL_PATH):
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained( MODEL_DIR , do_lower_case=True)
    
    #4) Preprocess the data:

    print( "tokenizing..."  )
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    
    #add special tokens
    tokenized_texts = [['[CLS]'] + sentence[:  (config.max_position_embeddings-2)  ] + [ "[SEP]" ] for sentence in         tokenized_texts ]
    
    #convert to ids+padding
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=config.max_position_embeddings, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)
    
    # create test tensors
    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_data = TensorDataset(prediction_inputs, prediction_masks)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=BATCH_SIZE)

    
    print( "start inference..."  )

    ## Prediction on test set
    # Put model in evaluation mode
    model.eval()
    # Tracking variables
    predictions_label , predictions_proba = [], []
    # Predict 
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(DEVICE) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask = batch
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
        # Forward pass, calculate logit predictions
            logits = model(b_input_ids, attention_mask=b_input_mask)
        # Move logits and labels to CPU
        logits = logits[0] #.detach().cpu().numpy()
        # Store predictions and true labels        
        prediction_proba =  F.softmax(logits, dim=1).detach().cpu().numpy()
        predictions_proba.append(  prediction_proba )
        
    flat_predictions_proba = [item for sublist in predictions_proba for item in sublist]
    predictions_labels = np.argmax(flat_predictions_proba, axis=1).flatten()
        
    os.makedirs(  os.path.dirname( OUTPUT_FILE ) , exist_ok=True  )
    
    with open( OUTPUT_FILE ,  "w"  ) as fp:
        for pred_label, pred_proba_label in zip(predictions_labels, flat_predictions_proba):
            fp.write( f"{pred_label} {pred_proba_label.tolist()}\n"     )