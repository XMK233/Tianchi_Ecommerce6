import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast as BertTokenizer
import numpy as np
from config import Config

def load_train_data():
    reviews = pd.read_csv(Config.TRAIN_REVIEWS)
    labels = pd.read_csv(Config.TRAIN_LABELS)
    labels = labels.fillna('_')
    return reviews, labels

def load_test_data():
    reviews = pd.read_csv(Config.TEST_REVIEWS)
    return reviews

def get_spans_from_tags(tags, text, offset_mapping):
    # tags: list of int
    # text: original string
    # offset_mapping: list of (start, end)
    
    aspects = []
    opinions = []
    
    i = 0
    while i < len(tags):
        tag = tags[i]
        # B-ASP
        if tag == Config.TAG2IDX['B-ASP']:
            start_idx = i
            i += 1
            while i < len(tags) and tags[i] == Config.TAG2IDX['I-ASP']:
                i += 1
            end_idx = i - 1
            
            # Map to char offsets
            char_start = offset_mapping[start_idx][0]
            char_end = offset_mapping[end_idx][1]
            if char_start != 0 or char_end != 0: # Valid span
                span_text = text[char_start:char_end]
                aspects.append((span_text, char_start, char_end))
            continue
            
        # B-OP
        if tag == Config.TAG2IDX['B-OP']:
            start_idx = i
            i += 1
            while i < len(tags) and tags[i] == Config.TAG2IDX['I-OP']:
                i += 1
            end_idx = i - 1
            
            char_start = offset_mapping[start_idx][0]
            char_end = offset_mapping[end_idx][1]
            if char_start != 0 or char_end != 0:
                span_text = text[char_start:char_end]
                opinions.append((span_text, char_start, char_end))
            continue
            
        i += 1
    
    return aspects, opinions

class ExtractionDataset(Dataset):
    def __init__(self, reviews_df, labels_df=None, tokenizer=None, max_len=512, is_train=False):
        self.reviews = reviews_df
        self.labels_df = labels_df # store labels_df
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train # Add is_train flag
        self.label_map = {}
        
        if labels_df is not None:
            for _, row in labels_df.iterrows():
                rid = row['id']
                if rid not in self.label_map:
                    self.label_map[rid] = []
                
                # Check column existence safely
                if 'AspectTerms' in row and row['AspectTerms'] != '_':
                    try:
                         # Ensure these columns exist in labels_df
                         if 'A_start' in row and 'A_end' in row:
                            self.label_map[rid].append((int(row['A_start']), int(row['A_end']), 'ASP'))
                    except: pass
                
                if 'OpinionTerms' in row and row['OpinionTerms'] != '_':
                    try:
                        if 'O_start' in row and 'O_end' in row:
                            self.label_map[rid].append((int(row['O_start']), int(row['O_end']), 'OP'))
                    except: pass

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        # Use iloc for integer indexing
        row = self.reviews.iloc[idx]
        text = str(row['Reviews'])
        rid = row['id']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        # Squeeze to remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        offset_mapping = item.pop('offset_mapping')
        
        # Data Augmentation: Random Masking (Token Masking)
        # Apply only during training
        if self.is_train:
            input_ids = item['input_ids']
            seq_len = input_ids.size(0)
            mask_prob = 0.15
            rand = torch.rand(seq_len)
            
            # Create mask: probability < mask_prob and not special tokens
            # 101=[CLS], 102=[SEP], 0=[PAD] for BERT
            # Check tokenizer specific ids if needed, but these are standard for BERT
            mask_indices = (rand < mask_prob) & (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
            
            # Replace with [MASK] (token id 103)
            input_ids[mask_indices] = 103 
            item['input_ids'] = input_ids

        # Label generation logic (only if labels are provided)
        # Note: We need labels for training
        if self.labels_df is not None:
             # Initialize tags with 'O'
            tags = [Config.TAG2IDX['O']] * self.max_len
            spans = self.label_map.get(rid, [])
            
            # Convert offset_mapping to list for iteration
            offset_mapping_list = offset_mapping.tolist()
            
            for start_char, end_char, type_ in spans:
                start_token = -1
                end_token = -1
                
                # Find token range for the span
                for i, (off_s, off_e) in enumerate(offset_mapping_list):
                    # Skip special tokens or padding which usually have (0,0) or similar
                    if off_s == 0 and off_e == 0 and i != 0: continue 
                    
                    # Check if token is within the character span
                    # This logic aligns tokens to character spans
                    if off_e > start_char and off_s < end_char:
                         if start_token == -1: start_token = i
                         end_token = i
                
                # If a valid token range is found, assign tags
                if start_token != -1:
                    if type_ == 'ASP':
                        tags[start_token] = Config.TAG2IDX['B-ASP']
                        if end_token > start_token:
                            for k in range(start_token+1, end_token+1):
                                tags[k] = Config.TAG2IDX['I-ASP']
                    elif type_ == 'OP':
                        tags[start_token] = Config.TAG2IDX['B-OP']
                        if end_token > start_token:
                            for k in range(start_token+1, end_token+1):
                                tags[k] = Config.TAG2IDX['I-OP']
            
            item['labels'] = torch.tensor(tags, dtype=torch.long)
        
        return item, text, rid, offset_mapping

class ClassificationDataset(Dataset):
    def __init__(self, reviews_df, samples, tokenizer, max_len=512, is_train=False):
        self.reviews_df = reviews_df.set_index('id')
        self.samples = samples # List of dicts
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_train = is_train
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        rid = sample['id']
        asp = sample['aspect']
        op = sample['opinion']
        
        # Retrieve review text
        if rid in self.reviews_df.index:
            text = str(self.reviews_df.loc[rid, 'Reviews'])
        else:
            text = "" # Should not happen ideally
        
        asp_text = asp if asp is not None and asp != '_' else '隐式'
        op_text = op if op is not None and op != '_' else ''
        
        # Prepare text pair for classification
        # Format: [CLS] Review [SEP] Aspect [SEP] Opinion [SEP] ?? or similar
        # Standard BERT pair: text_a, text_b
        # Here text_a = Review, text_b = Aspect [SEP] Opinion
        text_b = f"{asp_text}[SEP]{op_text}"
        
        encoding = self.tokenizer(
            text,
            text_b,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Data Augmentation: Random Masking (Token Masking)
        if self.is_train:
            input_ids = item['input_ids']
            seq_len = input_ids.size(0)
            mask_prob = 0.15
            rand = torch.rand(seq_len)
            
            # Masking logic
            # Avoid [CLS](101), [SEP](102), [PAD](0)
            mask_indices = (rand < mask_prob) & (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
            
            input_ids[mask_indices] = 103
            item['input_ids'] = input_ids
        
        # Add labels if available
        if 'category' in sample:
            cat_idx = Config.CAT2IDX.get(sample['category'], 0) # 0 is usually '无' or background if defined, check Config
            pol_idx = Config.POL2IDX.get(sample['polarity'], 0)
            item['cat_labels'] = torch.tensor(cat_idx, dtype=torch.long)
            item['pol_labels'] = torch.tensor(pol_idx, dtype=torch.long)
            
        return item
