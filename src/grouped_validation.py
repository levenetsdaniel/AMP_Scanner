"""Сравнение случайной и групповой validation без обращения к меткам test."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .experiment import (SEED, device_for, load_frames, peptide_loader,
                         train_transformer, transformer_pass)
from .features_extractor import extract_features
from .peptide_transformer import PeptideTransformer


def nearest_similarity(train, val):
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False)
    references = vectorizer.fit_transform(pd.concat([train.seq, val.seq]))
    index = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
    index.fit(references[:len(train)])
    distances, _ = index.kneighbors(references[len(train):], n_neighbors=1)
    return 1 - distances[:, 0]


def report_predictions(frame, probabilities, split, model):
    labels = frame.y_func.to_numpy()
    lengths = frame.seq.str.len().to_numpy()
    charges = np.array([(seq.count("K") + seq.count("R")
                         - seq.count("D") - seq.count("E")) / len(seq)
                        for seq in frame.seq])
    predictions = (probabilities >= 0.5).astype(int)
    rows = [{"split": split, "model": model, "subset": "all", "n": len(frame),
             "positives": int(labels.sum()), "roc_auc": roc_auc_score(labels, probabilities),
             "average_precision": average_precision_score(labels, probabilities),
             "recall_at_0.5": recall_score(labels, predictions)}]
    for lower, upper in ((51, 70), (71, 100)):
        subset = (lengths >= lower) & (lengths <= upper)
        positive = subset & (labels == 1)
        low_charge = positive & (charges <= 0.05)
        for name, mask in ((f"AMP_{lower}_{upper}", positive),
                           (f"AMP_{lower}_{upper}_charge_le_0.05", low_charge)):
            rows.append({"split": split, "model": model, "subset": name,
                         "n": int(mask.sum()), "positives": int(mask.sum()),
                         "roc_auc": np.nan, "average_precision": np.nan,
                         "recall_at_0.5": recall_score(labels[mask], predictions[mask])
                         if mask.any() else np.nan})
    details = pd.DataFrame({"split": split, "model": model, "name": frame["name"],
                            "length": lengths, "charge_density": charges,
                            "y_true": labels, "score": probabilities,
                            "y_pred_at_0.5": predictions})
    return rows, details


def run(epochs=15, batch_size=32, threshold=0.6, output=None, device_name="auto"):
    output = output or Path("outputs/grouped_validation")
    output.mkdir(parents=True, exist_ok=True)
    summaries, predictions, split_rows = [], [], []
    for split in ("random", "grouped"):
        train, val, _ = load_frames("pepanno", split=split,
                                    similarity_threshold=threshold)
        similarity = nearest_similarity(train, val)
        split_rows.append({"split": split, "train_n": len(train), "val_n": len(val),
                           "train_positive": int(train.y_func.sum()),
                           "val_positive": int(val.y_func.sum()),
                           "nearest_similarity_mean": similarity.mean(),
                           "nearest_similarity_max": similarity.max(),
                           "val_charge_le_0.05_AMP_51_100": int(sum(
                               val.y_func.eq(1) & val.seq.str.len().between(51, 100)
                               & val.seq.map(lambda s: (s.count("K") + s.count("R")
                                 - s.count("D") - s.count("E")) / len(s) <= 0.05)))})

        x_train = extract_features(train, "seq")
        x_val = extract_features(val, "seq")
        y_train = train.y_func.to_numpy()
        models = {
            "logreg": make_pipeline(StandardScaler(),
                                     LogisticRegression(C=0.1, max_iter=3000)),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1),
        }
        for name, model in models.items():
            model.fit(x_train, y_train)
            score = model.predict_proba(x_val)[:, 1]
            rows, details = report_predictions(val, score, split, name)
            summaries.extend(rows)
            predictions.append(details)

        checkpoint_dir = output / split
        train_transformer(train, val, checkpoint_dir, epochs, batch_size, device_name)
        saved = torch.load(checkpoint_dir / "transformer.pt", map_location="cpu",
                           weights_only=True)
        device = device_for(device_name)
        transformer = PeptideTransformer(**saved["config"]).to(device)
        transformer.load_state_dict(saved["state_dict"])
        _, labels, score = transformer_pass(
            transformer, peptide_loader(val, batch_size), device)
        if not np.array_equal(labels, val.y_func.to_numpy()):
            raise RuntimeError("Порядок validation изменился")
        rows, details = report_predictions(val, score, split, "transformer")
        summaries.extend(rows)
        predictions.append(details)

    pd.DataFrame(split_rows).to_csv(output / "splits.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "metrics.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "predictions.csv", index=False)
    print(pd.DataFrame(split_rows).round(4).to_string(index=False))
    print(pd.DataFrame(summaries).round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--similarity-threshold", type=float, default=0.6)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("outputs/grouped_validation"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.similarity_threshold < 1:
        parser.error("Проверьте epochs, batch-size и similarity-threshold")
    run(args.epochs, args.batch_size, args.similarity_threshold, args.output, args.device)


if __name__ == "__main__":
    main()
