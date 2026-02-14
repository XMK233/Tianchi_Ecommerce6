import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast as BertTokenizer, get_linear_schedule_with_warmup
from config import Config
from utils_loc import load_train_data, ExtractionDataset, ClassificationDataset
from model import ExtractionModel, ClassificationModel
import os
import random
import numpy as np
from tqdm import tqdm

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_extraction(reviews_df, labels_df, tokenizer, device):
    print("Training Extraction Model...")
    dataset = ExtractionDataset(reviews_df, labels_df, tokenizer, Config.MAX_LEN)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model = ExtractionModel.from_pretrained(Config.BERT_PATH, num_labels=len(Config.TAGS))
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=Config.LR)
    total_steps = len(loader) * Config.EPOCHS_EXTRACT
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    model.train()
    for epoch in range(Config.EPOCHS_EXTRACT):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            batch_data = batch[0]
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            token_type_ids = batch_data['token_type_ids'].to(device)
            labels = batch_data['labels'].to(device)
            
            model.zero_grad()
            loss, _ = model(input_ids, attention_mask, token_type_ids, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        print(f"Loss: {total_loss/len(loader)}")
        
    torch.save(model.state_dict(), os.path.join(Config.TRAINED_MODEL_PATH, 'extraction_model.pth'))
    return model

def prepare_classification_samples(reviews_df, labels_df):
    samples = []
    # Group labels by ID
    grouped = labels_df.groupby('id')
    
    for rid in reviews_df['id']:
        if rid not in grouped.groups: continue
        group = grouped.get_group(rid)
        
        # Collect GT pairs
        gt_pairs = set()
        aspects = set()
        opinions = set()
        
        for _, row in group.iterrows():
            asp = row['AspectTerms'] if row['AspectTerms'] != '_' else None
            op = row['OpinionTerms']
            cat = row['Categories']
            pol = row['Polarities']
            
            samples.append({
                'id': rid,
                'aspect': asp,
                'opinion': op,
                'category': cat,
                'polarity': pol
            })
            
            gt_pairs.add((asp, op))
            if asp is not None: aspects.add(asp)
            opinions.add(op)
            
        # Generate Negatives
        # 1. Explicit Aspect + Explicit Opinion (not paired)
        for a in aspects:
            for o in opinions:
                if (a, o) not in gt_pairs:
                    samples.append({
                        'id': rid,
                        'aspect': a,
                        'opinion': o,
                        'category': 'None',
                        'polarity': '中性' # Placeholder
                    })
        
        # 2. Implicit Aspect + Explicit Opinion (not paired)
        # If (None, o) is not in GT, add it as negative
        for o in opinions:
            if (None, o) not in gt_pairs:
                 samples.append({
                        'id': rid,
                        'aspect': None,
                        'opinion': o,
                        'category': 'None',
                        'polarity': '中性'
                    })
                    
    return samples

def train_classification(reviews_df, labels_df, tokenizer, device):
    print("Training Classification Model...")
    samples = prepare_classification_samples(reviews_df, labels_df)
    # Shuffle samples
    random.shuffle(samples)
    
    dataset = ClassificationDataset(reviews_df, samples, tokenizer, Config.MAX_LEN)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    model = ClassificationModel.from_pretrained(Config.BERT_PATH)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=Config.LR)
    total_steps = len(loader) * Config.EPOCHS_CLASSIFY
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    model.train()
    for epoch in range(Config.EPOCHS_CLASSIFY):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            cat_labels = batch['cat_labels'].to(device)
            pol_labels = batch['pol_labels'].to(device)
            
            model.zero_grad()
            loss, _, _ = model(input_ids, attention_mask, token_type_ids, cat_labels, pol_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        print(f"Loss: {total_loss/len(loader)}")
        
    torch.save(model.state_dict(), os.path.join(Config.TRAINED_MODEL_PATH, 'classification_model.pth'))
    return model

if __name__ == '__main__':
    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)
    reviews, labels = load_train_data()
    
    train_extraction(reviews, labels, tokenizer, device)
    train_classification(reviews, labels, tokenizer, device)
