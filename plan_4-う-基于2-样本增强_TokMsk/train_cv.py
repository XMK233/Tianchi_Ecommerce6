import os
import random
import numpy as np
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast as BertTokenizer, get_linear_schedule_with_warmup

from config import Config
from utils_loc import load_train_data, ExtractionDataset, ClassificationDataset
from model import ExtractionModel, ClassificationModel
from train import prepare_classification_samples


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_folds(ids, num_folds: int = 5, seed: int = 42):
    ids = [int(x) for x in ids]
    rng = np.random.RandomState(seed)
    rng.shuffle(ids)

    fold_sizes = [len(ids) // num_folds] * num_folds
    for i in range(len(ids) % num_folds):
        fold_sizes[i] += 1

    folds = []
    offset = 0
    for sz in fold_sizes:
        folds.append(ids[offset : offset + sz])
        offset += sz
    return folds


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _train_extraction_one_fold(reviews_df, labels_df, tokenizer, device, fold_dir: str, epochs: int):
    # Pass is_train=True for token masking augmentation
    dataset = ExtractionDataset(reviews_df, labels_df, tokenizer, Config.MAX_LEN, is_train=True)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = ExtractionModel.from_pretrained(Config.BERT_PATH, num_labels=len(Config.TAGS))
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LR)
    total_steps = max(1, len(loader) * max(1, epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    print(f"  Training Extraction Model (Fold: {os.path.basename(fold_dir)})")
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"    Epoch {epoch+1}/{epochs}"):
            batch_data = batch[0]
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)
            labels = batch_data["labels"].to(device)

            model.zero_grad()
            loss, _ = model(input_ids, attention_mask, token_type_ids, labels=labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        print(f"    Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(fold_dir, "extraction_model.pth"))


def _train_classification_one_fold(reviews_df, labels_df, tokenizer, device, fold_dir: str, epochs: int):
    samples = prepare_classification_samples(reviews_df, labels_df)
    random.shuffle(samples)

    dataset = ClassificationDataset(reviews_df, samples, tokenizer, Config.MAX_LEN, is_train=True)
    loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)

    model = ClassificationModel.from_pretrained(Config.BERT_PATH)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LR)
    total_steps = max(1, len(loader) * max(1, epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    model.train()
    print(f"  Training Classification Model (Fold: {os.path.basename(fold_dir)})")
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"    Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            cat_labels = batch["cat_labels"].to(device)
            pol_labels = batch["pol_labels"].to(device)

            model.zero_grad()
            loss, _, _ = model(input_ids, attention_mask, token_type_ids, cat_labels, pol_labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        print(f"    Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), os.path.join(fold_dir, "classification_model.pth"))


def train_all_folds(num_folds: int = 5, seed: int = 42, output_subdir: str = "cv",
                    epochs_extract: int = None, epochs_classify: int = None):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(Config.BERT_PATH)

    reviews_df, labels_df = load_train_data()
    all_ids = reviews_df["id"].astype(int).tolist()
    folds = make_folds(all_ids, num_folds=num_folds, seed=seed)

    base_dir = os.path.join(Config.TRAINED_MODEL_PATH, output_subdir)
    _ensure_dir(base_dir)

    if epochs_extract is None:
        epochs_extract = Config.EPOCHS_EXTRACT
    if epochs_classify is None:
        epochs_classify = Config.EPOCHS_CLASSIFY

    print(f"Starting {num_folds}-fold CV training.")
    print(f"Seed: {seed}")
    print(f"Epochs (Extract): {epochs_extract}")
    print(f"Epochs (Classify): {epochs_classify}")
    print(f"Output Directory: {base_dir}")

    for fold_idx in range(num_folds):
        print(f"Training Fold {fold_idx + 1}/{num_folds}...")
        val_ids = set(folds[fold_idx])
        train_ids = [rid for rid in all_ids if rid not in val_ids]

        fold_dir = os.path.join(base_dir, f"fold_{fold_idx}")
        _ensure_dir(fold_dir)

        train_reviews = reviews_df[reviews_df["id"].astype(int).isin(train_ids)].reset_index(drop=True)
        train_labels = labels_df[labels_df["id"].astype(int).isin(train_ids)].reset_index(drop=True)

        _train_extraction_one_fold(
            train_reviews, train_labels, tokenizer, device, fold_dir=fold_dir, epochs=epochs_extract
        )
        _train_classification_one_fold(
            train_reviews, train_labels, tokenizer, device, fold_dir=fold_dir, epochs=epochs_classify
        )

    return base_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ABSA models with Cross Validation")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of folds for CV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs_extract", type=int, default=None, help="Epochs for extraction model")
    parser.add_argument("--epochs_classify", type=int, default=None, help="Epochs for classification model")
    parser.add_argument("--output_subdir", type=str, default="cv", help="Subdirectory to save models")

    args = parser.parse_args()

    train_all_folds(
        num_folds=args.num_folds,
        seed=args.seed,
        output_subdir=args.output_subdir,
        epochs_extract=args.epochs_extract,
        epochs_classify=args.epochs_classify
    )
