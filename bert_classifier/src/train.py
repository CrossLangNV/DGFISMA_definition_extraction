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

import torch

from transformers import BertPreTrainedModel, BertConfig, BertModel, AdamW

from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer

from transformers import DistilBertConfig, DistilBertModel
from transformers.modeling_distilbert import DistilBertPreTrainedModel

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from models import BertSequenceClassifier, DistilBertSequenceClassifier, DistilBertGRUSequenceClassifier
from utils import csv_field_limit
     
csv_field_limit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #Input-output:
    parser.add_argument("--filename", dest="FILENAME",
                        help='Path to the training data. A csv with at each line: "plain text"￭label_tag￭label_nr. The plain text should be enclosed in ".."', required=True)
    parser.add_argument("--delimiter", dest="DELIMITER",
                        help="delimiter used in input csv", required=False, type=str, default="￭")
    parser.add_argument("--model_storage_directory", dest="MODEL_STORAGE_DIRECTORY",
                        help="output directory", required=True)
    #Preprocessing parameters:
    parser.add_argument("--train_size", dest="TRAIN_SIZE",
                        help="Fraction of data used for training. 1-TRAIN_SIZE will be used for validation", required=False, type=float, default=0.8 )
    parser.add_argument("--seed", dest="SEED",
                        help="seed used for train/val split", required=False, type=int, default=2018)
    #neural network parameters:
    parser.add_argument("--output_dim", dest="OUTPUT_DIM",
                        help="Nr of output classes. Defaults to the binary case", required=False, type=int, default=2)
    parser.add_argument('--bert_model', dest="BERT_MODEL", help='bert model chosen: bert or distilbert architecture, e.g. "bert-base-uncased", "distilbert-base-uncased",...', default='distilbert-base-uncased' )
    parser.add_argument('--train_bert', dest="TRAIN_BERT", help='whether to train all layers of BERT, freeze all layers of BERT, or only train last layer of BERT', choices=[ 'freeze_all_layers' , 'last_layer' , 'all_layers' ] ,  default='all_layers' )
    parser.add_argument('--rnn_model', dest='RNN_MODEL', help='send hidden states of BERT to RNN instead of using the BERT pooling layers. Only available in combination with distilbert' , action='store_true')
    parser.add_argument('--rnn_bidirectional', dest='RNN_BIDIRECTIONAL', help='Bidirectional rnn layer. Only used when --rnn_model is used, otherwise ignored.' , action='store_true')
    parser.add_argument("--rnn_hidden_dim", dest="RNN_HIDDEN_DIM",
                        help="Hidden dimension of the RNN. Only used when --rnn_model is used, otherwise ignored.", required=False, type=int, default=256)
    parser.add_argument("--rnn_n_layers", dest="RNN_N_LAYERS",
                        help="Number of RNN layers. Only used when --rnn_model is used, otherwise ignored.", required=False, type=int, default=2)
    parser.add_argument("--rnn_dropout", dest="RNN_DROPOUT",
                        help="Dropout on the rnn layer if n_layers>2. Only used when --rnn_model is used, otherwise ignored.", required=False, type=float, default=0.25)
    parser.add_argument("--batch_size", dest="BATCH_SIZE",
                        help="batch size", required=False, type=int, default=32)
    parser.add_argument("--epochs", dest="EPOCHS",
                        help="nr of training epochs", required=False, type=int, default=10)
    parser.add_argument("--learning_rate", dest="LEARNING_RATE",
                        help="initial learning rate of adam optimizer", required=False, type=float, default=2e-5)
    #device:
    parser.add_argument("--device", dest="DEVICE",
                        help="device: cuda:0, cuda:1,...cpu", required=False, type=str, default="cpu")
    args = parser.parse_args()

    #1) Parameters:
    FILENAME=args.FILENAME 
    DELIMITER=args.DELIMITER #delimiter of the input csv
    MODEL_STORAGE_DIRECTORY=args.MODEL_STORAGE_DIRECTORY
    DEVICE =args.DEVICE  #"cuda:0", cpu,..

    TRAIN_SIZE=args.TRAIN_SIZE
    VAL_SIZE=1.0-TRAIN_SIZE
    SEED=args.SEED

    #neural network parameters
    OUTPUT_DIM=args.OUTPUT_DIM
    BERT_MODEL=args.BERT_MODEL
    TRAIN_BERT=args.TRAIN_BERT
    RNN_MODEL=args.RNN_MODEL
    if RNN_MODEL:
        RNN_BIDIRECTIONAL=args.RNN_BIDIRECTIONAL
        RNN_HIDDEN_DIM=args.RNN_HIDDEN_DIM 
        RNN_N_LAYERS=args.RNN_N_LAYERS
        RNN_DROPOUT=args.RNN_DROPOUT
   
    BATCH_SIZE=args.BATCH_SIZE
    EPOCHS=args.EPOCHS
    LEARNING_RATE=args.LEARNING_RATE

    #2) Set run specific envirorment configurations

    timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    model_directory = os.path.join(MODEL_STORAGE_DIRECTORY, timestamp) #directory
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
    
    #3) Set up config:
    
    bert_model_path=BERT_MODEL

    if 'distil' in bert_model_path:
        ArchitectureConfig=DistilBertConfig
    else:
        ArchitectureConfig=BertConfig

    if os.path.exists( bert_model_path  ):
        if os.path.exists(os.path.join(bert_model_path, CONFIG_NAME)):
            print( f"loading {bert_model_path}" )
            config = ArchitectureConfig.from_json_file(os.path.join(bert_model_path, CONFIG_NAME))
        elif os.path.exists(os.path.join(bert_model_path, 'bert_config.json')):
            print( f"loading {bert_model_path}" )
            config = ArchitectureConfig.from_json_file(os.path.join(bert_model_path, 'bert_config.json'))
        else:
            raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
    else:
        config = ArchitectureConfig.from_pretrained(bert_model_path )
        
    config.__setattr__( 'num_labels', OUTPUT_DIM )

    if RNN_MODEL:
    #only necessary for GRUDistilBertClassifier
        config.__setattr__( 'rnn_bidirectional', RNN_BIDIRECTIONAL   )
        config.__setattr__( 'rnn_hidden_dim', RNN_HIDDEN_DIM   )
        config.__setattr__( 'rnn_n_layers', RNN_N_LAYERS  )
        config.__setattr__( 'rnn_dropout',  RNN_DROPOUT )
    
    #3) Load the data:
    dataset=pd.read_csv( FILENAME, delimiter=DELIMITER, quoting=csv.QUOTE_NONE, header=None, engine='python', names=['text', 'label_tag','label' ])
    
    labels=dataset.label.tolist()
    sentences=dataset.text.tolist()
    sentences=[ sentence[1:-1] for sentence in sentences  ]  #sentences are enclosed in " "
    
    #4) Preprocess the data:
    if 'distil' in bert_model_path:
        tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path, do_lower_case=True)
    else:
        tokenizer = BertTokenizer.from_pretrained( bert_model_path , do_lower_case=True)

    log.info( "tokenizing..."  )
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
    
    # Use train_test_split to split our data into train and validation sets for training
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                                random_state=SEED, test_size=VAL_SIZE)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                 random_state=SEED, test_size=VAL_SIZE)

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
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE)

    #4) set-up the model:
    
    if 'distil' in bert_model_path:
        if RNN_MODEL:
            model = DistilBertGRUSequenceClassifier.from_pretrained( bert_model_path , config=config  ) 
        else:    
            model = DistilBertSequenceClassifier.from_pretrained( bert_model_path , config=config )

    else:
        model = BertSequenceClassifier.from_pretrained( bert_model_path , config=config  ) 

    model.to(DEVICE)

    
    if TRAIN_BERT=='freeze_all_layers':
        model.freeze_bert_encoder()  
    elif TRAIN_BERT=='last_layer':
        model.freeze_bert_encoder()  
        model.unfreeze_bert_encoder_last_layers()
    elif TRAIN_BERT=='all_layers':
        pass
    
    #5) train the model:
    
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, pred_flat)  #np.sum(pred_flat == labels_flat) / len(labels_flat)
 
    def train_func( epochs, model, optimizer, train_dataloader, validation_dataloader, device ):
        
        max_grad_norm=1.0
        train_loss_set=[]

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping is not in AdamW anymore (so you can use amp without issue)
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
            
            log.info(f"epoch: {epoch+1}"  )
            log.info("Train loss: {}".format(tr_loss/nb_tr_steps)) 
            log.info("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
            log.info("\n" )
         
        return tr_loss
        
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False  )  #to reproduce BertAdam specific behavior set correct_bias=False
    
    tr_loss=train_func( EPOCHS, model, optimizer, train_dataloader, validation_dataloader, DEVICE )
    
    #7)Save the model:
    log.info( f"saving model in {model_directory}/{BERT_MODEL}_model_{EPOCHS}.pth"  )
    #save the config file
    config.save_pretrained( model_directory )
    #save the tokenizer
    tokenizer.save_pretrained( model_directory ) 
    #save the model
    torch.save({
                'epoch': EPOCHS,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': tr_loss,
                'learning_rate': LEARNING_RATE
                },  os.path.join(model_directory, f"{BERT_MODEL}_model_{EPOCHS}.pth"  ) )
    
    