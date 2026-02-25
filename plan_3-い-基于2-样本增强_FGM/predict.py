import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertTokenizerFast as BertTokenizer
from config import Config
from utils_loc import load_test_data, ExtractionDataset, ClassificationDataset, get_spans_from_tags
from model import ExtractionModel, ClassificationModel
import pandas as pd
from tqdm import tqdm
import os

def predict():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)
    
    # Load Data
    test_reviews = load_test_data()
    
    # --- Step 1: Extraction ---
    print("Running Extraction...")
    extract_dataset = ExtractionDataset(test_reviews, None, tokenizer, Config.MAX_LEN)
    extract_loader = DataLoader(extract_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    extract_model = ExtractionModel.from_pretrained(Config.BERT_PATH, num_labels=len(Config.TAGS))
    extract_model.load_state_dict(torch.load(os.path.join(Config.TRAINED_MODEL_PATH, 'extraction_model.pth')))
    extract_model.to(device)
    extract_model.eval()
    
    extracted_data = {} # rid -> {'aspects': [], 'opinions': []}
    
    with torch.no_grad():
        for batch in tqdm(extract_loader):
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
                # Mapping comes as tensor in loader? No, I returned list in utils?
                # In utils: return item, text, rid, offset_mapping (list)
                # But DataLoader collates. List of lists?
                # Let's check collate_fn. Default collate handles lists.
                # Actually, offset_mapping is a list of lists.
                
                # Check mapping structure
                # mapping is a list of [start, end]
                
                # Check lengths
                if len(tags) != len(mapping):
                    print(f"Error: tags len {len(tags)}, mapping len {len(mapping)}")
                    # Truncate tags or pad mapping?
                    # Usually mapping should be 512.
                    # If mapping is shorter, it's weird.
                    # Just skip or truncate tags.
                    min_len = min(len(tags), len(mapping))
                    tags = tags[:min_len]
                
                aspects, opinions = get_spans_from_tags(tags, text, mapping)
                extracted_data[rid] = {'aspects': aspects, 'opinions': opinions}
                
    # --- Step 2: Classification ---
    print("Running Classification...")
    classify_model = ClassificationModel.from_pretrained(Config.BERT_PATH)
    classify_model.load_state_dict(torch.load(os.path.join(Config.TRAINED_MODEL_PATH, 'classification_model.pth')))
    classify_model.to(device)
    classify_model.eval()
    
    # Prepare classification samples
    samples = []
    sample_indices = [] # to map back to (rid, aspect, opinion)
    
    for rid, data in extracted_data.items():
        aspects = data['aspects'] # list of (text, start, end)
        opinions = data['opinions']
        
        # Candidate pairs
        # 1. Explicit Aspect + Explicit Opinion
        for asp in aspects:
            for op in opinions:
                samples.append({
                    'id': rid,
                    'aspect': asp[0],
                    'opinion': op[0]
                })
                sample_indices.append((rid, asp[0], op[0], 'Explicit'))
        
        # 2. Implicit Aspect + Explicit Opinion
        for op in opinions:
            samples.append({
                'id': rid,
                'aspect': None,
                'opinion': op[0]
            })
            sample_indices.append((rid, '_', op[0], 'Implicit'))
            
    if not samples:
        print("No opinions found! Generating empty result.")
        # Handle empty result...
        return

    classify_dataset = ClassificationDataset(test_reviews, samples, tokenizer, Config.MAX_LEN)
    classify_loader = DataLoader(classify_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    results = []
    
    idx = 0
    with torch.no_grad():
        for batch in tqdm(classify_loader):
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
                    rid, asp, op, type_ = sample_indices[idx]
                    results.append({
                        'id': rid,
                        'AspectTerms': asp,
                        'OpinionTerms': op,
                        'Categories': category,
                        'Polarities': polarity
                    })
                idx += 1
                
    # Format Output
    # The requirement is: ID should be ascending. 
    # If ID has no prediction, fill with _.
    # Columns: id, AspectTerms, OpinionTerms, Categories, Polarities
    # Note: README says "Result.csv". Format: ID, AspectTerm, OpinionTerm, Category, Polarity
    
    output_df = pd.DataFrame(results)
    
    # Handle missing IDs
    all_ids = test_reviews['id'].unique()
    final_rows = []
    
    if not output_df.empty:
        grouped = output_df.groupby('id')
    else:
        grouped = {}
        
    for rid in sorted(all_ids):
        if not output_df.empty and rid in grouped.groups:
            group = grouped.get_group(rid)
            for _, row in group.iterrows():
                final_rows.append(row.to_dict())
        else:
            final_rows.append({
                'id': rid,
                'AspectTerms': '_',
                'OpinionTerms': '_',
                'Categories': '_',
                'Polarities': '_'
            })
            
    final_df = pd.DataFrame(final_rows)
    final_df = final_df[['id', 'AspectTerms', 'OpinionTerms', 'Categories', 'Polarities']]
    final_df.to_csv(Config.TEST_OUTPUT, index=False, header=False) # README says "不需要表头" (No header)? 
    # README: "提交的文件中不需要表头部分" -> Yes, header=False.
    
    print(f"Saved results to {Config.TEST_OUTPUT}")

if __name__ == '__main__':
    predict()
