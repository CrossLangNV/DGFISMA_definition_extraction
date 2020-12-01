import torch
from torch import nn
from transformers import BertPreTrainedModel, BertConfig, DistilBertConfig, DistilBertModel, BertModel
from transformers.modeling_distilbert import DistilBertPreTrainedModel


class BertSequenceClassifier(BertPreTrainedModel):
    """
    Reimplementation of sequence classifier based on BERT
    """

    def __init__(self, bert_model_config: BertConfig):
        super(BertSequenceClassifier, self).__init__(bert_model_config)
        self.num_labels = bert_model_config.num_labels

        self.bert = BertModel(bert_model_config)
        self.dropout = nn.Dropout(p=bert_model_config.hidden_dropout_prob)
        self.classifier = nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels)

        self.init_weights()

    # input_ids, token_type_ids, attention_masks
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):

        bert_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = bert_outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,)  # + bert_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.bert.named_parameters():
            if "encoder.layer.11" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.bert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


class DistilBertSequenceClassifier(DistilBertPreTrainedModel):
    """
    Reimplementation of sequence classifier based on BERT
    """

    def __init__(self, bert_model_config: DistilBertConfig):
        super(DistilBertSequenceClassifier, self).__init__(bert_model_config)
        self.num_labels = bert_model_config.num_labels

        self.distilbert = DistilBertModel(bert_model_config)
        self.pre_classifier = nn.Linear(bert_model_config.hidden_size, bert_model_config.hidden_size)
        self.classifier = nn.Linear(bert_model_config.hidden_size, bert_model_config.num_labels)
        self.dropout = nn.Dropout(p=bert_model_config.dropout)

        self.init_weights()

    # input_ids, token_type_ids, attention_masks
    def forward(self, input_ids=None, attention_mask=None, labels=None):

        bert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = bert_outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)

        logits = self.classifier(pooled_output)

        outputs = (logits,)  # + bert_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def freeze_bert_encoder(self):
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.distilbert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.distilbert.named_parameters():
            if "layer.5" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.distilbert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True


# see:
# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6%20-%20Transformers%20for%20Sentiment%20Analysis.ipynb
class DistilBertGRUSequenceClassifier(DistilBertPreTrainedModel):
    """
    Reimplementation of sequence classifier based on BERT. With GRU. 
    """

    def __init__(self, bert_model_config: DistilBertConfig):
        super(DistilBertGRUSequenceClassifier, self).__init__(bert_model_config)
        self.num_labels = bert_model_config.num_labels

        self.rnn_bidirectional = bert_model_config.rnn_bidirectional
        self.rnn_hidden_dim = bert_model_config.rnn_hidden_dim
        self.rnn_n_layers = bert_model_config.rnn_n_layers
        self.rnn_dropout = bert_model_config.rnn_dropout

        self.distilbert = DistilBertModel(bert_model_config)

        self.rnn = nn.GRU(bert_model_config.hidden_size,
                          self.rnn_hidden_dim,
                          num_layers=self.rnn_n_layers,
                          bidirectional=self.rnn_bidirectional,
                          batch_first=True,
                          dropout=0 if self.rnn_n_layers < 2 else self.rnn_dropout)

        # self.classifier=nn.Linear( 256, bert_model_config.num_labels )

        self.out = nn.Linear(self.rnn_hidden_dim * 2 if self.rnn_bidirectional else self.rnn_hidden_dim,
                             bert_model_config.num_labels)

        self.dropout = nn.Dropout(p=bert_model_config.dropout)

        self.init_weights()

    # input_ids, token_type_ids, attention_masks
    def forward(self, input_ids=None, attention_mask=None, labels=None):

        # bert_outputs=self.bert(  input_ids=input_ids , token_type_ids=token_type_ids, attention_mask=attention_mask )
        bert_outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)

        hidden_state = bert_outputs[0]  # (bs, seq_len, bert_hid_dim), e.g. ( 32, 512, 768 )

        _, hidden = self.rnn(hidden_state)

        # hidden = [n layers * n directions, bs, rnn_hidden_dim], e.g. ( 1*1, 32, 256 )

        if self.rnn_bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

            # hidden = [bs, rnn_hidden_dim]

        logits = self.out(hidden)

        outputs = (logits,)  # + bert_outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = torch.nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

    def freeze_bert_encoder(self):
        for param in self.distilbert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.distilbert.parameters():
            param.requires_grad = True

    def unfreeze_bert_encoder_last_layers(self):
        for name, param in self.distilbert.named_parameters():
            if "layer.5" in name or "pooler" in name:
                param.requires_grad = True

    def unfreeze_bert_encoder_pooler_layer(self):
        for name, param in self.distilbert.named_parameters():
            if "pooler" in name:
                param.requires_grad = True
