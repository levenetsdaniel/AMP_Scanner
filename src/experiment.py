import argparse
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .features_extractor import extract_features
from .load_dataset import load_data
from .peptide_dataset import PeptideDataset, VOCAB, collate_peptides
from .peptide_transformer import PeptideTransformer


SEED = 42
ESM_NAME = "facebook/esm2_t6_8M_UR50D"
OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"
MODEL_CONFIG = {"vocab_size": len(VOCAB) + 1, "d_model": 128, "nhead": 4,
                "num_layers": 2, "max_len": 512, "dropout": 0.1}


def load_frames(dataset="amplify"):
    raw_train, test = load_data(dataset)
    train_source = raw_train.drop_duplicates(subset="seq").reset_index(drop=True)
    train, val = train_test_split(train_source, test_size=0.2,
                                  random_state=SEED, stratify=train_source["y_func"])

    return train.reset_index(drop=True), val.reset_index(drop=True), test


def best_threshold(y_true, probabilities):
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)

    return float(thresholds[np.argmax(f1)])


def metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    return {
        "roc_auc": roc_auc_score(y_true, probabilities),
        "average_precision": average_precision_score(y_true, probabilities),
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "mcc": matthews_corrcoef(y_true, predicted),
        "threshold": threshold, "n": len(y_true), "positives": int(sum(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def fit_best(candidates, x_train, y_train, x_val, y_val):
    best = None

    for description, model in candidates:
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, probabilities)
        if best is None or auc > best[0]:
            best = (auc, description, model, probabilities)

    _, description, model, probabilities = best
    return {"model": model, "threshold": best_threshold(y_val, probabilities),
            "selection": description}


def train_baselines(train, val, output):
    output.mkdir(parents=True, exist_ok=True)
    x_train = extract_features(train, "seq")
    x_val = extract_features(val, "seq")
    y_train = train["y_func"].to_numpy()
    y_val = val["y_func"].to_numpy()

    candidates = {
        "logreg": [(f"C={c}", make_pipeline(StandardScaler(),
                    LogisticRegression(C=c, max_iter=3000))) for c in (0.01, 0.1, 1.0, 10.0)],
        "random_forest": [(f"max_depth={depth}", RandomForestClassifier(
            n_estimators=200, max_depth=depth, random_state=SEED, n_jobs=-1))
                          for depth in (None, 12)],
    }

    for name, models in candidates.items():
        saved = fit_best(models, x_train, y_train, x_val, y_val)
        joblib.dump(saved, output / f"{name}.joblib")
        prob = saved["model"].predict_proba(x_val)[:, 1]
        print(name, saved["selection"], metrics(y_val, prob, saved["threshold"]), '/n')


def device_for(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def peptide_loader(frame, batch_size, shuffle=False):
    return DataLoader(PeptideDataset(frame), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=collate_peptides, num_workers=0)

def transformer_pass(model, loader, device, optimizer=None, description=None):
    training = optimizer is not None
    model.train(training)
    loss_sum, count = 0.0, 0
    labels, probabilities = [], []
    batches = tqdm(loader, desc=description, unit="batch", leave=False, mininterval=1.0,
                   disable=description is None)

    for batch in batches:
        x = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        y = batch["labels"].to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(x, mask)
            loss = nn.functional.cross_entropy(logits, y)

            if training:
                loss.backward()
                optimizer.step()

        loss_sum += loss.item() * len(y)
        count += len(y)
        labels.extend(y.cpu().tolist())
        probabilities.extend(torch.softmax(logits.detach(), dim=1)[:, 1].cpu().tolist())

    return loss_sum / count, np.asarray(labels), np.asarray(probabilities)


def train_transformer(train, val, output, epochs=15, batch_size=32, device_name="auto"):
    output.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = device_for(device_name)
    model = PeptideTransformer(**MODEL_CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    train_loader = peptide_loader(train, batch_size, shuffle=True)
    val_loader = peptide_loader(val, batch_size)

    best_auc = -1.0

    progress = tqdm(range(1, epochs + 1), desc="Transformer", unit="epoch")

    for epoch in progress:
        train_loss, _, _ = transformer_pass(model, train_loader, device, optimizer,
                                            description=f"Epoch {epoch} train")
        val_loss, y_val, prob = transformer_pass(model, val_loader, device,
                                                description=f"Epoch {epoch} val")

        auc = roc_auc_score(y_val, prob)
        progress.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{val_loss:.4f}",
                             val_auc=f"{auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save({"state_dict": model.state_dict(), "config": MODEL_CONFIG,
                        "threshold": best_threshold(y_val, prob), "best_epoch": epoch},
                       output / "transformer.pt")


def load_esm(device):
    tokenizer = AutoTokenizer.from_pretrained(ESM_NAME)
    model = AutoModel.from_pretrained(ESM_NAME, add_pooling_layer=False).to(device)
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def esm_embeddings(model, tokenizer, sequences, batch_size, device, description="ESM-2"):
    vectors = []
    starts = tqdm(range(0, len(sequences), batch_size), desc=description,
                  unit="batch", leave=False, mininterval=1.0)

    for start in starts:
        tokens = tokenizer(sequences[start:start + batch_size], padding=True,
                           return_special_tokens_mask=True, return_tensors="pt")
        residue_mask = tokens["attention_mask"].bool() & ~tokens["special_tokens_mask"].bool()
        hidden = model(input_ids=tokens["input_ids"].to(device),
                       attention_mask=tokens["attention_mask"].to(device)).last_hidden_state
        mask = residue_mask.to(device).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        vectors.append(pooled.float().cpu().numpy())

    return np.concatenate(vectors)


def train_esm(train, val, output, batch_size=16, device_name="auto"):
    output.mkdir(parents=True, exist_ok=True)

    device = device_for(device_name)
    esm, tokenizer = load_esm(device)

    x_train = esm_embeddings(esm, tokenizer, train["seq"].tolist(), batch_size, device,
                             description="ESM-2 train")
    x_val = esm_embeddings(esm, tokenizer, val["seq"].tolist(), batch_size, device,
                           description="ESM-2 validation")

    saved = fit_best([(f"C={c}", make_pipeline(StandardScaler(),
                      LogisticRegression(C=c, max_iter=3000))) for c in (0.01, 0.1, 1.0, 10.0)],
                     x_train, train["y_func"].to_numpy(), x_val, val["y_func"].to_numpy())

    joblib.dump(saved, output / "esm2_logreg.joblib")
    prob = saved["model"].predict_proba(x_val)[:, 1]

    print("esm2_logreg", saved["selection"],
          metrics(val["y_func"].to_numpy(), prob, saved["threshold"]), '/n')


def evaluate(test, output, batch_size=32, device_name="auto"):
    y = test["y_func"].to_numpy()
    predictions, rows = [], []
    features = extract_features(test, "seq")

    device = device_for(device_name)

    for name in ("logreg", "random_forest", "transformer", "esm2_logreg"):
        if name in ("logreg", "random_forest"):
            saved = joblib.load(output / f"{name}.joblib")
            prob = saved["model"].predict_proba(features)[:, 1]

        elif name == "transformer":
            saved = torch.load(output / "transformer.pt", map_location="cpu", weights_only=True)
            model = PeptideTransformer(**saved["config"]).to(device)
            model.load_state_dict(saved["state_dict"])
            _, labels, prob = transformer_pass(model, peptide_loader(test, batch_size), device,
                                               description="Transformer test")

            if not np.array_equal(labels, y):
                raise ValueError("Порядок строк test изменился")

        else:
            saved = joblib.load(output / "esm2_logreg.joblib")
            esm, tokenizer = load_esm(device)
            vectors = esm_embeddings(esm, tokenizer, test["seq"].tolist(), batch_size, device,
                                     description="ESM-2 test")
            prob = saved["model"].predict_proba(vectors)[:, 1]

        threshold = saved["threshold"]
        rows.append({"model": name, **metrics(y, prob, threshold)})
        predictions.append(pd.DataFrame({"name": test["name"], "model": name,
                                         "y_true": y, "score": prob,
                                         "y_pred": (prob >= threshold).astype(int)}))

    output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(output / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    print(report.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run",
                        choices=["run", "train-baselines", "train-transformer", "train-esm", "evaluate"])
    parser.add_argument("--dataset", choices=["amplify", "pepanno"], default="amplify")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")

    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs и --batch-size должны быть положительными")

    output = args.output or OUTPUT_ROOT / ("simple" if args.dataset == "amplify" else "pepanno")
    train, val, test = load_frames(args.dataset)

    if args.command in ("run", "train-baselines"):
        train_baselines(train, val, output)

    if args.command in ("run", "train-transformer"):
        train_transformer(train, val, output, args.epochs, args.batch_size, args.device)

    if args.command in ("run", "train-esm"):
        train_esm(train, val, output, args.batch_size, args.device)

    if args.command in ("run", "evaluate"):
        evaluate(test, output, args.batch_size, args.device)


if __name__ == "__main__":
    main()
