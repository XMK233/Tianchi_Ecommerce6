import os
import torch
import argparse
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast as BertTokenizer
from tqdm import tqdm

from config import Config
from utils_loc import load_train_data, ExtractionDataset, ClassificationDataset, get_spans_from_tags
from model import ExtractionModel, ClassificationModel
from train_cv import make_folds, train_all_folds


def _fold_dir(base_dir: str, fold_idx: int) -> str:
    return os.path.join(base_dir, f"fold_{fold_idx}")


def _models_exist(base_dir: str, fold_idx: int) -> bool:
    fold_dir = _fold_dir(base_dir, fold_idx)
    return os.path.exists(os.path.join(fold_dir, "extraction_model.pth")) and os.path.exists(
        os.path.join(fold_dir, "classification_model.pth")
    )


def _load_models(tokenizer, device, fold_dir: str):
    extract_model = ExtractionModel.from_pretrained(Config.BERT_PATH, num_labels=len(Config.TAGS))
    extract_model.load_state_dict(torch.load(os.path.join(fold_dir, "extraction_model.pth"), map_location="cpu"))
    extract_model.to(device)
    extract_model.eval()

    classify_model = ClassificationModel.from_pretrained(Config.BERT_PATH)
    classify_model.load_state_dict(torch.load(os.path.join(fold_dir, "classification_model.pth"), map_location="cpu"))
    classify_model.to(device)
    classify_model.eval()

    return extract_model, classify_model


def _predict_one_fold(reviews_df, labels_df, fold_ids, tokenizer, device, fold_dir: str):
    fold_ids = [int(x) for x in fold_ids]
    val_reviews = reviews_df[reviews_df["id"].astype(int).isin(fold_ids)].reset_index(drop=True)
    val_labels = labels_df[labels_df["id"].astype(int).isin(fold_ids)].reset_index(drop=True)

    extract_model, classify_model = _load_models(tokenizer, device, fold_dir=fold_dir)

    extract_dataset = ExtractionDataset(val_reviews, None, tokenizer, Config.MAX_LEN)
    extract_loader = DataLoader(extract_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    extracted_data = {}
    with torch.no_grad():
        for batch in tqdm(extract_loader, desc=f"Fold extraction ({os.path.basename(fold_dir)})"):
            batch_data = batch[0]
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)

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

                if isinstance(mapping, torch.Tensor):
                    mapping = mapping.tolist()

                if len(tags) != len(mapping):
                    min_len = min(len(tags), len(mapping))
                    tags = tags[:min_len]
                    mapping = mapping[:min_len]

                aspects, opinions = get_spans_from_tags(tags, text, mapping)
                extracted_data[rid] = {"aspects": aspects, "opinions": opinions}

    samples = []
    sample_indices = []
    for rid, data in extracted_data.items():
        aspects = data["aspects"]
        opinions = data["opinions"]

        for asp in aspects:
            for op in opinions:
                samples.append({"id": rid, "aspect": asp[0], "opinion": op[0]})
                sample_indices.append((rid, asp[0], op[0]))

        for op in opinions:
            samples.append({"id": rid, "aspect": None, "opinion": op[0]})
            sample_indices.append((rid, "_", op[0]))

    predicted_tuples = []
    if samples:
        classify_dataset = ClassificationDataset(val_reviews, samples, tokenizer, Config.MAX_LEN)
        classify_loader = DataLoader(classify_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

        idx = 0
        with torch.no_grad():
            for batch in tqdm(classify_loader, desc=f"Fold classification ({os.path.basename(fold_dir)})"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)

                cat_logits, pol_logits = classify_model(input_ids, attention_mask, token_type_ids)
                cat_preds = torch.argmax(cat_logits, dim=1).cpu().numpy()
                pol_preds = torch.argmax(pol_logits, dim=1).cpu().numpy()

                for i in range(len(cat_preds)):
                    cat_idx = cat_preds[i]
                    pol_idx = pol_preds[i]

                    category = Config.IDX2CAT[cat_idx]
                    polarity = Config.IDX2POL[pol_idx]

                    if category != "None":
                        rid, asp, op = sample_indices[idx]
                        predicted_tuples.append((rid, asp, op, category, polarity))
                    idx += 1

    gt_set = set()
    for _, row in val_labels.iterrows():
        rid = int(row["id"])
        asp = row["AspectTerms"]
        op = row["OpinionTerms"]
        cat = row["Categories"]
        pol = row["Polarities"]
        gt_set.add((rid, asp, op, cat, pol))

    pred_set = set(predicted_tuples)
    return gt_set, pred_set


def evaluate_cv(num_folds: int = 5, seed: int = 42, model_subdir: str = "cv", train_if_missing: bool = True,
                epochs_extract: int = None, epochs_classify: int = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)

    reviews_df, labels_df = load_train_data()
    all_ids = reviews_df["id"].astype(int).tolist()
    folds = make_folds(all_ids, num_folds=num_folds, seed=seed)

    base_dir = os.path.join(Config.TRAINED_MODEL_PATH, model_subdir)

    if train_if_missing:
        need_train = False
        for fold_idx in range(num_folds):
            if not _models_exist(base_dir, fold_idx):
                need_train = True
                break
        if need_train:
            print("Missing models detected. Starting training...")
            train_all_folds(num_folds=num_folds, seed=seed, output_subdir=model_subdir,
                            epochs_extract=epochs_extract, epochs_classify=epochs_classify)

    global_gt = set()
    global_pred = set()

    for fold_idx in range(num_folds):
        fold_dir = _fold_dir(base_dir, fold_idx)
        gt_set, pred_set = _predict_one_fold(
            reviews_df, labels_df, folds[fold_idx], tokenizer, device, fold_dir=fold_dir
        )
        global_gt |= gt_set
        global_pred |= pred_set

        s = len(gt_set.intersection(pred_set))
        p = len(pred_set)
        g = len(gt_set)
        precision = s / p if p > 0 else 0
        recall = s / g if g > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print("-" * 30)
        print(f"Fold {fold_idx}")
        print(f"Correct (S): {s}")
        print(f"Predicted (P): {p}")
        print(f"Ground Truth (G): {g}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")

    S = len(global_gt.intersection(global_pred))
    P = len(global_pred)
    G = len(global_gt)
    precision = S / P if P > 0 else 0
    recall = S / G if G > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("=" * 30)
    print("CV Overall")
    print(f"Correct (S): {S}")
    print(f"Predicted (P): {P}")
    print(f"Ground Truth (G): {G}")
    print("=" * 30)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ABSA models with Cross Validation")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_subdir", type=str, default="cv", help="Subdirectory to load models from")
    parser.add_argument("--train_if_missing", type=int, default=1, help="Train if models are missing (1=True, 0=False)")
    parser.add_argument("--epochs_extract", type=int, default=None, help="Epochs for extraction model (if training needed)")
    parser.add_argument("--epochs_classify", type=int, default=None, help="Epochs for classification model (if training needed)")

    args = parser.parse_args()

    evaluate_cv(
        num_folds=args.num_folds,
        seed=args.seed,
        model_subdir=args.model_subdir,
        train_if_missing=bool(args.train_if_missing),
        epochs_extract=args.epochs_extract,
        epochs_classify=args.epochs_classify
    )
