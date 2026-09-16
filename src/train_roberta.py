"""
src/train_roberta.py

Trains a RoBERTa-base model for the 8-class or 3-class strategy prediction task. 
Trains and evaluates strictly on train/val splits, leaving the test set untouched.

Defaults to the optimal hyperparameters from the DistilBERT sweep. Final 
checkpoints use speaker-tagged inputs, making this the only tagged-input 
model in the repository. Custom overrides save to isolated subfolders.

Usage:
    python train_roberta.py --task 8class
    python train_roberta.py --task 8class --lr 1e-5 --weight_decay 0.05
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import (
    load_and_clean,
    build_8class_examples,
    build_3class_examples,
    flatten_context,
    apply_conversation_level_split,
    STRATEGY_LABELS,
    COARSE_LABELS,
)

# --- Fixed config, matching the original DistilBERT winning recipe -----------

MODEL_NAME = "roberta-base"
MAX_SEQ_LEN = 96
BATCH_SIZE = 16
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
MAX_EPOCHS = 12
PATIENCE = 2
MIN_DELTA = 0.001
SEED = 42

# DistilBERT-winning LR per task, reused here unchanged (see report_notes.md)
FIXED_LR = {
    "8class": 3e-5,
    "3class": 2e-5,
}

TASKS = {
    "8class": {
        "build_examples": build_8class_examples,
        "labels": STRATEGY_LABELS,
        "label_key": "gold_strategy_id",
    },
    "3class": {
        "build_examples": build_3class_examples,
        "labels": COARSE_LABELS,
        "label_key": "gold_coarse_id",
    },
}


def set_seed(seed=SEED):
    """Seeds Python, NumPy and torch (CPU + CUDA). cudnn determinism is not forced,
    so runs are seeded but not bit-identical on GPU.

    Called by: main().
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class StrategyDataset(Dataset):
    """Training dataset: renders each context to speaker-tagged text once, then
    tokenises lazily per item. Tags are hardcoded on -- the final checkpoints are
    the only tagged-input models in this repo.

    Called by: main(), for the train and val sets.
    """

    def __init__(self, examples, label_key, tokenizer, max_length=MAX_SEQ_LEN):
        """Stores the rendered texts and integer labels; no tokenising yet.

        Called by: main(), when building the train/val datasets.
        """
        # speaker tags on ("[seeker]"/"[supporter]" prefixes) -- see report_notes.md
        self.texts = [flatten_context(ex["context"], include_speaker_tags=True) for ex in examples]
        self.labels = [ex[label_key] for ex in examples]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Number of training examples -- what the DataLoader iterates over.

        Called by: the DataLoader, to size the epoch.
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """Tokenises one example into fixed-length (96,) input_ids and
        attention_mask plus a scalar label.

        Called by: the DataLoader, once per item per epoch.
        """
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def compute_class_weights(labels, n_classes):
    """Balanced class weights, n_samples / (n_classes * count). A class at exactly
    its even share gets 1.0; rarer classes get more. Fed to CrossEntropyLoss.

    Called by: main().
    """
    counts = np.bincount(labels, minlength=n_classes)
    n_samples = len(labels)
    weights = n_samples / (n_classes * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float)


def run_epoch(model, loader, device, optimizer=None, scheduler=None, loss_fn=None):
    """One pass over a loader, training or evaluating depending on whether an
    optimizer was passed. Returns (average loss, macro-F1).

    Called by: main().
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            total_loss += loss.item() * input_ids.size(0)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(all_labels)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


def main():
    """Fine-tunes the model for one task: builds the split, trains with early
    stopping on validation macro-F1, and saves the best checkpoint. Test is never
    touched here.

    Called by: the __main__ guard at the bottom of the file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["8class", "3class"], required=True)
    parser.add_argument("--lr", type=float, default=None,
                         help="Override the DistilBERT-winning LR (see FIXED_LR). "
                              "Use to test whether RoBERTa's larger capacity needs a lower LR.")
    parser.add_argument("--weight_decay", type=float, default=None,
                         help="Override WEIGHT_DECAY (default 0.01). E.g. 0.05, DistilBERT's best "
                              "regularization-grid value, to test combined with a lower LR.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type != "cuda":
        print("WARNING: no GPU detected -- this will be very slow.")

    task_cfg = TASKS[args.task]
    lr = args.lr if args.lr is not None else FIXED_LR[args.task]
    weight_decay = args.weight_decay if args.weight_decay is not None else WEIGHT_DECAY
    set_seed(SEED)

    convs = load_and_clean()
    examples = task_cfg["build_examples"](convs)
    n_classes = len(task_cfg["labels"])
    label_key = task_cfg["label_key"]

    train_examples, val_examples, test_examples = apply_conversation_level_split(convs, examples, seed=SEED)
    print(f"Train: {len(train_examples)}  Val: {len(val_examples)}  Test: {len(test_examples)} (test unused by this script)")
    is_default_recipe = (args.lr is None and args.weight_decay is None)
    print(f"Model: {MODEL_NAME}  lr={lr}"
          f"{' (DistilBERT-winning LR, reused unchanged)' if args.lr is None else ' (override)'}"
          f"  wd={weight_decay}{'' if args.weight_decay is None else ' (override)'}")

    # default recipe keeps the flat path; any override gets its own subfolder
    if is_default_recipe:
        output_root = Path(__file__).resolve().parent.parent / "outputs" / args.task / "roberta"
    else:
        run_name = f"lr{lr:.0e}_wd{weight_decay}".replace("+", "").replace("-0", "-")
        output_root = Path(__file__).resolve().parent.parent / "outputs" / args.task / "roberta" / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    metrics_path = output_root / "metrics.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=n_classes).to(device)

    train_ds = StrategyDataset(train_examples, label_key, tokenizer)
    val_ds = StrategyDataset(val_examples, label_key, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    train_labels = [ex[label_key] for ex in train_examples]
    class_weights = compute_class_weights(train_labels, n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * MAX_EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    training_start = time.time()  # wall-clock, for the report's cost/time table

    with open(metrics_path, "w") as f:
        for epoch in range(1, MAX_EPOCHS + 1):
            epoch_start = time.time()
            train_loss, train_f1 = run_epoch(model, train_loader, device, optimizer, scheduler, loss_fn)
            val_loss, val_f1 = run_epoch(model, val_loader, device, loss_fn=loss_fn)
            epoch_seconds = time.time() - epoch_start

            record = {"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1,
                      "val_loss": val_loss, "val_macro_f1": val_f1, "epoch_seconds": round(epoch_seconds, 1)}
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_macro_f1={val_f1:.4f}  ({epoch_seconds:.1f}s)")

            if val_f1 > best_val_f1 + MIN_DELTA:
                best_val_f1 = val_f1
                epochs_without_improvement = 0
                model.save_pretrained(output_root / "best_model")
                tokenizer.save_pretrained(output_root / "best_model")
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    print(f"  early stop: no improvement > {MIN_DELTA} for {PATIENCE} epochs")
                    break

    total_seconds = time.time() - training_start
    print(f"\n=== {args.task} RoBERTa-base result (lr={lr}, wd={weight_decay}): "
          f"best val macro-F1 = {best_val_f1:.4f} ===")
    print(f"Total training wall-clock: {total_seconds:.0f}s ({total_seconds/60:.1f} min) -- for the report's cost/time table.")
    print(f"Saved to: {output_root / 'best_model'}")
    print(f"Compare against DistilBERT's {'0.2312' if args.task == '8class' else '0.4348'} "
          f"(see report_notes.md), RoBERTa's own first run (0.2110, lr=3e-5/wd=0.01, 8class), "
          f"and the published RoBERTa baseline (25.04 macro-F1, 8-class only, EmoDynamiX paper).")


if __name__ == "__main__":
    main()
