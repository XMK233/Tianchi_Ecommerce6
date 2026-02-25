import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel
from config import Config

class ExtractionModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, len(Config.TAGS))
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only calculate loss on active parts (ignore padding which is usually 0, but here 'O' is 0)
            # Actually attention_mask handles padding in BERT, but for loss we need to ignore padded tokens
            # Usually we set padded labels to -100
            # But in my Utils, I set them to 'O' (0). 
            # It's better to use attention mask to mask out loss.
            
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, len(Config.TAGS))[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)
            
        return (loss, logits) if loss is not None else logits

class ClassificationModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        
        self.cat_classifier = nn.Linear(config.hidden_size, len(Config.CATEGORIES))
        self.pol_classifier = nn.Linear(config.hidden_size, len(Config.POLARITIES))
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, cat_labels=None, pol_labels=None):
        outputs = self.bert(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids
        )
        pooled_output = outputs[1] # [CLS]
        pooled_output = self.dropout(pooled_output)
        
        cat_logits = self.cat_classifier(pooled_output)
        pol_logits = self.pol_classifier(pooled_output)
        
        loss = None
        if cat_labels is not None and pol_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            cat_loss = loss_fct(cat_logits, cat_labels)
            pol_loss = loss_fct(pol_logits, pol_labels)
            loss = cat_loss + pol_loss
            
        return (loss, cat_logits, pol_logits) if loss is not None else (cat_logits, pol_logits)
