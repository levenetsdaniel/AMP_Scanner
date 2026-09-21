from pathlib import Path
import random

import numpy as np
import torch

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader

from load_dataset import load_data
from peptide_dataset import PeptideDataset, VOCAB, collate_peptides
from peptide_transformer import PeptideTransformer


SEED = 42
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts" / "custom_transformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

def clean_train(df):
    df = df.copy()
    df["seq"] = df["seq"].str.strip().str.upper()

    conflicts = df.groupby("seq")["y_func"].nunique()
    bad_sequences = conflicts[conflicts > 1].index

    df = df[~df["seq"].isin(bad_sequences)]
    df = df.drop_duplicates(subset="seq")

    return df.reset_index(drop=True)


def make_loader(df, shuffle):
    return DataLoader(
        PeptideDataset(df),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate_peptides,
        num_workers=0,
    )


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)

    losses = []
    probabilities = []
    labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, y)

            if training:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())

        probabilities.extend(
            torch.softmax(logits.detach(), dim=1)[:, 1]
            .cpu()
            .numpy()
            .tolist()
        )

        labels.extend(y.cpu().numpy().tolist())

    return (
        float(np.mean(losses)),
        roc_auc_score(labels, probabilities),
    )


def main():
    raw_train, _ = load_data()
    df = clean_train(raw_train)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["y_func"],
    )

    train_loader = make_loader(train_df, shuffle=True)
    val_loader = make_loader(val_df, shuffle=False)

    model = PeptideTransformer(
        vocab_size=len(VOCAB) + 1,
        d_model=128,
        nhead=4,
        num_layers=2,
        max_len=512,
        dropout=0.1,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01,
    )

    best_auc = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_auc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        with torch.no_grad():
            val_loss, val_auc = run_epoch(
                model,
                val_loader,
                criterion,
            )

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train AUC: {train_auc:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_auc": best_auc,
                    "seed": SEED,
                    "train_indices": train_df.index.to_list(),
                    "val_indices": val_df.index.to_list(),
                },
                OUTPUT_DIR / "best_model.pt",
            )

            print("Saved best model")

    print("Best validation ROC-AUC:", best_auc)


if __name__ == "__main__":
    main()