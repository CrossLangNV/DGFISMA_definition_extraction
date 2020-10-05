import re
import os
import json
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertPreTrainedModel, BertConfig, BertModel, AdamW

from transformers.tokenization_bert import BertTokenizer
from transformers.tokenization_distilbert import DistilBertTokenizer

from transformers import DistilBertConfig, DistilBertModel
from transformers.modeling_distilbert import DistilBertPreTrainedModel

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

import fasttext

from bert_classifier.src.models import BertSequenceClassifier, DistilBertSequenceClassifier, DistilBertGRUSequenceClassifier


class DefinitionFinder():
    
    def __init__( self, sentences=None):
        
        '''
        :param sentences: List. List of Strings (i.e. sentences).
        '''
        
        if sentences is None:
            self.sentences=[]
        else:
            self.sentences=list( sentences )
            
        self.definitions=[]    
                
    def get_definitions_regex( self ):
        
        '''
        Method checks, for each sentence in self.sentences, if it is a definition, using a regex. It returns a list of Booleans. 
        
        :return: List. List of Booleans. 
        '''

        self.definitions=[]
        
        regex=r"\‘[a-z0-9 \-(){}_]+\’.{,20}(\bmeans\b|\bshall mean\b|in relation to\b)"  # maximum 20 chars between detected term ‘ ’ and "means", "shall mean", ...

        for sentence in self.sentences:
        
            match=re.search( regex, sentence, re.IGNORECASE )

            if match:
                definition_bool=True
            else:
                definition_bool=False
            
            self.definitions.append( definition_bool )
        
        #sanity check:
        assert( len( self.sentences) == len(self.definitions) )   
        
        return self.definitions
    
    def get_definitions_bert( self, model_path, device="cuda:0" , batch_size=32, rnn_model=False, nr_of_threads=6 ):
        
        '''
        Method checks, for each sentence in self.sentences, if it is a definition, using a Bert-based text classification model. It returns a list of Booleans.

        :param model_path: String. Path to the pretrained Bert-based text classification model (e.g. ../distilbert-base-uncased_model_10.pth.
        :param device: String. Device used for inference. Can be "cuda:0", "cuda:1", "cpu". Default is "cuda:0".
        :param batch_size: Int. Batch size used for inference. Default value is 32. 
        :param rnn_model: Boolean. Whether the pretrained model sends hidden states of BERT to an RNN instead of using the BERT pooling layers. Only available in combination with a distilbert model. Default is False.
        :nr_of_threads. Nr of threads used for inference on cpu. Ignored when gpu is used.
        :return: List. List of Booleans. 
        '''
        
        self.definitions=[]
                
        #1) Parameters:
        MODEL_PATH=model_path
        DEVICE=device  #"cuda:0", cpu,..
        RNN_MODEL=rnn_model
        BATCH_SIZE=batch_size

        MODEL_DIR=os.path.dirname(  MODEL_PATH )
        
        torch.set_num_threads( nr_of_threads )

        #2) Load the config, model and tokenizer

        if os.path.exists( MODEL_PATH  ):
            if os.path.exists(os.path.join( MODEL_DIR , 'config.json' )):
                print( f"loading {MODEL_PATH}" )
                with open( os.path.join( MODEL_DIR, "config.json" ) ) as json_file:
                    config = json.load(json_file)
                if config['model_type'] == "distilbert":
                    ArchitectureConfig=DistilBertConfig
                elif config['model_type'] == "bert":
                    ArchitectureConfig=BertConfig
                else:
                     raise ValueError( f"config.json located at {MODEL_DIR} is not a valid configuration file. Model type not supported or not available" )
                config = ArchitectureConfig.from_json_file(os.path.join(MODEL_DIR, 'config.json' ))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else: 
            raise ValueError( f"Cannot find BERT based model at {MODEL_PATH} you are attempting to load." )

        #model:

        if isinstance (config, DistilBertConfig):
            if RNN_MODEL:
                model = DistilBertGRUSequenceClassifier( config )
            else:    
                model = DistilBertSequenceClassifier( config )
        elif isinstance (config, BertConfig):
            model = BertSequenceClassifier( config  )
        else:
            raise ValueError( f"config.json located at {MODEL_DIR} is not a valid configuration file." )

        checkpoint = torch.load( MODEL_PATH , map_location=torch.device( DEVICE )  )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)

        #tokenizer
        if isinstance (config, DistilBertConfig):
            tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR, do_lower_case=True)
        elif isinstance ( config, BertConfig):
            tokenizer = BertTokenizer.from_pretrained( MODEL_DIR , do_lower_case=True)
        else:
            raise ValueError( f"config.json located at {MODEL_DIR} is not a valid configuration file." )

        #4) Preprocess the data:

        print( "tokenizing..."  )
        tokenized_texts = [tokenizer.tokenize(sent) for sent in self.sentences]

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

        # create tensors
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

        self.definitions=[True if definition==1 else False for definition in predictions_labels ]
        
        #sanity check
        
        assert( len( self.sentences ) == len( self.definitions)   )
        
        return self.definitions  
    
    
    def get_definitions_fasttext( self, model_path, pos_label='__label__definition', neg_label='__label__no_definition' ):
        
        '''
        Method checks, for each sentence in self.sentences, if it is a definition, using a fastText text classification model. It returns a list of Booleans.

        :param model_path: String. Path to the pretrained fastText text classification model (e.g. ../model.bin.
        :param pos_label: String. Positive label. Label the fastText model is trained on. Default value '__label__definition'.
        :param neg_label: String. Negative label. Default value '__label__no_definition'
        :return: List. List of Booleans. 
        '''
        
        self.definitions=[]
        
        try:
            model = fasttext.load_model( model_path )
        except ValueError:
            raise ValueError( f"Could not load fastText model at {model_path}." )
        
        #sanity check:
        try:
            assert( set(model.labels) == set( [ pos_label , neg_label ] )  )
        except AssertionError:
            raise AssertionError(f'fastText model labels {model.labels} should be same as "pos_label: {pos_label}" and "neg_label: {neg_label}".'  )
            
        pred_labels=model.predict( self.sentences  )[0]
        self.definitions=[ True if label[0] == pos_label else False for label in pred_labels  ]  

        #sanity check:
        
        assert( len( self.sentences ) == len( self.definitions)   )

        return self.definitions
     
