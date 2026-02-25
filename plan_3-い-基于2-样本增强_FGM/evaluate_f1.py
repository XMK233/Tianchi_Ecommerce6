import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast as BertTokenizer
from config import Config
from utils_loc import load_train_data, ExtractionDataset, ClassificationDataset, get_spans_from_tags
from model import ExtractionModel, ClassificationModel
import pandas as pd
from tqdm import tqdm
import os
import numpy as np

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)
    
    # Load Data (Use Train Data for evaluation)
    print("Loading Train Data for Evaluation...")
    reviews_df, labels_df = load_train_data()
    
    # --- Step 1: Extraction Inference ---
    print("Running Extraction on Train Set...")
    # Note: We use the same extraction logic as predict.py
    extract_dataset = ExtractionDataset(reviews_df, None, tokenizer, Config.MAX_LEN)
    extract_loader = DataLoader(extract_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    extract_model = ExtractionModel.from_pretrained(Config.BERT_PATH, num_labels=len(Config.TAGS))
    extract_model.load_state_dict(torch.load(os.path.join(Config.TRAINED_MODEL_PATH, 'extraction_model.pth')))
    extract_model.to(device)
    extract_model.eval()
    
    extracted_data = {} # rid -> {'aspects': [], 'opinions': []}
    
    with torch.no_grad():
        for batch in tqdm(extract_loader, desc="Extraction"):
            batch_data = batch[0]
            input_ids = batch_data['input_ids'].to(device)
            attention_mask = batch_data['attention_mask'].to(device)
            token_type_ids = batch_data['token_type_ids'].to(device)
            
            logits = extract_model(input_ids, attention_mask, token_type_ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            preds = torch.argmax(logits, dim=2).cpu().numpy()
            
            texts = batch[1]
            rids = batch[2]
            offset_mappings = batch[3]
            
            for i, rid in enumerate(rids):
                rid = int(rid)
                tags = preds[i]
                text = texts[i]
                mapping = offset_mappings[i]
                
                # Fix length mismatch if any
                if len(tags) != len(mapping):
                    min_len = min(len(tags), len(mapping))
                    tags = tags[:min_len]
                    mapping = mapping[:min_len]
                
                aspects, opinions = get_spans_from_tags(tags, text, mapping)
                extracted_data[rid] = {'aspects': aspects, 'opinions': opinions}
                
    # --- Step 2: Classification Inference ---
    print("Running Classification on Candidate Pairs...")
    classify_model = ClassificationModel.from_pretrained(Config.BERT_PATH)
    classify_model.load_state_dict(torch.load(os.path.join(Config.TRAINED_MODEL_PATH, 'classification_model.pth')))
    classify_model.to(device)
    classify_model.eval()
    
    samples = []
    sample_indices = [] 
    
    for rid, data in extracted_data.items():
        aspects = data['aspects'] 
        opinions = data['opinions']
        
        # 1. Explicit Aspect + Explicit Opinion
        for asp in aspects:
            for op in opinions:
                samples.append({
                    'id': rid,
                    'aspect': asp[0],
                    'opinion': op[0]
                })
                sample_indices.append((rid, asp[0], op[0]))
        
        # 2. Implicit Aspect + Explicit Opinion
        for op in opinions:
            samples.append({
                'id': rid,
                'aspect': None,
                'opinion': op[0]
            })
            sample_indices.append((rid, '_', op[0]))

    if not samples:
        print("No candidates found.")
        return

    classify_dataset = ClassificationDataset(reviews_df, samples, tokenizer, Config.MAX_LEN)
    classify_loader = DataLoader(classify_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    predicted_tuples = [] # list of (id, aspect, opinion, category, polarity)
    
    idx = 0
    with torch.no_grad():
        for batch in tqdm(classify_loader, desc="Classification"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            
            cat_logits, pol_logits = classify_model(input_ids, attention_mask, token_type_ids)
            
            cat_preds = torch.argmax(cat_logits, dim=1).cpu().numpy()
            pol_preds = torch.argmax(pol_logits, dim=1).cpu().numpy()
            
            for i in range(len(cat_preds)):
                cat_idx = cat_preds[i]
                pol_idx = pol_preds[i]
                
                category = Config.IDX2CAT[cat_idx]
                polarity = Config.IDX2POL[pol_idx]
                
                if category != 'None':
                    rid, asp, op = sample_indices[idx]
                    predicted_tuples.append((rid, asp, op, category, polarity))
                idx += 1
                
    # --- Step 3: Compute F1 ---
    print("Computing F1 Score...")
    
    # 1. Build Ground Truth Set
    # Format: (id, AspectTerm, OpinionTerm, Category, Polarity)
    gt_set = set()
    for _, row in labels_df.iterrows():
        rid = row['id']
        asp = row['AspectTerms']
        op = row['OpinionTerms']
        cat = row['Categories']
        pol = row['Polarities']
        gt_set.add((rid, asp, op, cat, pol))
        
    # 2. Build Predicted Set
    pred_set = set(predicted_tuples)
    
    # 3. Calculate metrics
    S = len(gt_set.intersection(pred_set))
    P = len(pred_set)
    G = len(gt_set)
    
    precision = S / P if P > 0 else 0
    recall = S / G if G > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("-" * 30)
    print(f"Correct (S): {S}")
    print(f"Predicted (P): {P}")
    print(f"Ground Truth (G): {G}")
    print("-" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    
    # Optional: Save bad cases
    # Identify False Positives (in Pred but not GT)
    fp = pred_set - gt_set
    # Identify False Negatives (in GT but not Pred)
    fn = gt_set - pred_set
    
    with open('error_analysis.txt', 'w') as f:
        f.write("False Positives (Predicted but Wrong):\n")
        for item in list(fp)[:50]:
            f.write(str(item) + "\n")
        f.write("\nFalse Negatives (Missed):\n")
        for item in list(fn)[:50]:
            f.write(str(item) + "\n")
    print("Error analysis saved to error_analysis.txt")

if __name__ == "__main__":
    evaluate()
