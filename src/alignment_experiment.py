import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import Align
from Bio.Align import substitution_matrices
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from .experiment import (
    best_threshold, device_for, esm_embeddings, load_esm, load_frames, metrics,
    peptide_loader,
)
from .peptide_transformer import PeptideTransformer


OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "outputs"
K_VALUES = (1, 3, 5, 11)


def make_aligner(mode: str) -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner()
    aligner.mode = mode
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10
    aligner.extend_gap_score = -0.5
    return aligner


class DipeptideCandidateIndex:
    def __init__(self, sequences: list[str]):
        self.vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2), lowercase=False)
        vectors = self.vectorizer.fit_transform(sequences)
        self.neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        self.neighbors.fit(vectors)
        self.size = len(sequences)

    def query(self, sequences: list[str], count: int, batch_size: int = 128) -> np.ndarray:
        count = min(count, self.size)
        result = []

        for start in tqdm(range(0, len(sequences), batch_size), desc="Поиск кандидатов",
                          unit="batch", leave=False):
            vectors = self.vectorizer.transform(sequences[start:start + batch_size])
            result.append(self.neighbors.kneighbors(vectors, n_neighbors=count, return_distance=False))

        return np.vstack(result)


class EmbeddingCandidateIndex:
    def __init__(self, vectors: np.ndarray):
        self.neighbors = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        self.neighbors.fit(vectors)
        self.size = len(vectors)

    def query(self, vectors: np.ndarray, count: int, batch_size: int = 128) -> np.ndarray:
        count = min(count, self.size)
        result = []

        for start in tqdm(range(0, len(vectors), batch_size), desc="Поиск по эмбеддингам", unit="batch", leave=False):
            result.append(self.neighbors.kneighbors(vectors[start:start + batch_size], n_neighbors=count,  return_distance=False))

        return np.vstack(result)


@torch.inference_mode()
def transformer_embeddings(model: PeptideTransformer, frame: pd.DataFrame,
                           batch_size: int, device: torch.device,
                           description: str) -> np.ndarray:
    vectors = []
    for batch in tqdm(peptide_loader(frame, batch_size), desc=description, unit="batch", leave=False):
        embeddings = model.get_embeddings(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        vectors.append(embeddings.cpu().numpy())

    return np.concatenate(vectors)


def aligned_neighbors(queries: list[str], references: list[str],
                      candidate_ids: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    aligner = make_aligner(mode)
    reference_self = np.array([aligner.score(seq, seq) for seq in references])
    ranked_ids = np.empty_like(candidate_ids)
    ranked_scores = np.empty(candidate_ids.shape, dtype=np.float32)

    for row, (query, ids) in enumerate(tqdm(zip(queries, candidate_ids), total=len(queries),
                                            desc=f"Выравнивание ({mode})", unit="seq")):
        query_self = aligner.score(query, query)
        scores = np.array([
            max(0.0, aligner.score(query, references[index])) /
            np.sqrt(query_self * reference_self[index])
            for index in ids
        ])
        order = np.argsort(-scores, kind="stable")
        ranked_ids[row] = ids[order]
        ranked_scores[row] = scores[order]

    return ranked_ids, ranked_scores


def knn_probabilities(ids: np.ndarray, similarities: np.ndarray, reference_labels: np.ndarray, k: int) -> np.ndarray:
    if k < 1 or k > ids.shape[1]:
        raise ValueError("k должно быть от 1 до числа кандидатов")

    labels = reference_labels[ids[:, :k]]
    weights = similarities[:, :k]
    denominator = weights.sum(axis=1)
    weighted = (weights * labels).sum(axis=1)

    return np.divide(weighted, denominator, out=np.full(len(ids), reference_labels.mean()), where=denominator > 0)


def run_experiment(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
                   output: Path, candidate_count: int = 64,
                   modes: tuple[str, ...] = ("global", "local"),
                   retrieval: str = "esm2", batch_size: int = 16,
                   device_name: str = "auto", checkpoint: Path | None = None) -> pd.DataFrame:
    references = train["seq"].tolist()
    labels = train["y_func"].to_numpy()
    if len(references) < 2:
        raise ValueError("Для kNN нужны хотя бы две обучающие последовательности")

    if retrieval == "esm2":
        device = device_for(device_name)
        esm, tokenizer = load_esm(device)
        reference_vectors = esm_embeddings(esm, tokenizer, references, batch_size, device, description="ESM-2 reference")
        index = EmbeddingCandidateIndex(reference_vectors)
        val_vectors = esm_embeddings(esm, tokenizer, val["seq"].tolist(), batch_size, device, description="ESM-2 validation")
        val_candidates = index.query(val_vectors, candidate_count)

    elif retrieval == "transformer":
        if checkpoint is None or not checkpoint.is_file():
            raise FileNotFoundError("Для --retrieval transformer нужен обученный --checkpoint")
        device = device_for(device_name)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        transformer = PeptideTransformer(**saved["config"]).to(device)
        transformer.load_state_dict(saved["state_dict"])
        transformer.eval()
        reference_vectors = transformer_embeddings(transformer, train, batch_size, device,
                                                    "Transformer reference")
        index = EmbeddingCandidateIndex(reference_vectors)
        val_vectors = transformer_embeddings(transformer, val, batch_size, device,
                                              "Transformer validation")
        val_candidates = index.query(val_vectors, candidate_count)

    elif retrieval == "dipeptide":
        index = DipeptideCandidateIndex(references)
        val_candidates = index.query(val["seq"].tolist(), candidate_count)

    else:
        raise ValueError("retrieval должен быть esm2, transformer или dipeptide")

    candidate_count = min(candidate_count, len(references))
    k_values = [k for k in K_VALUES if k <= candidate_count]
    if not k_values:
        raise ValueError("Число кандидатов должно быть положительным")

    y_val = val["y_func"].to_numpy()
    best = None
    validation_rows = []

    for mode in modes:
        ids, scores = aligned_neighbors(val["seq"].tolist(), references, val_candidates, mode)

        for k in k_values:
            probability = knn_probabilities(ids, scores, labels, k)
            threshold = best_threshold(y_val, probability)
            result = metrics(y_val, probability, threshold)
            validation_rows.append({"retrieval": retrieval, "mode": mode, "k": k, "candidates": candidate_count, **result})

            if best is None or result["roc_auc"] > best["roc_auc"]:
                best = {"mode": mode, "k": k, "threshold": threshold, "roc_auc": result["roc_auc"]}

    if retrieval == "esm2":
        test_vectors = esm_embeddings(esm, tokenizer, test["seq"].tolist(), batch_size, device, description="ESM-2 test")
        test_candidates = index.query(test_vectors, candidate_count)

    elif retrieval == "transformer":
        test_vectors = transformer_embeddings(transformer, test, batch_size, device,
                                               "Transformer test")
        test_candidates = index.query(test_vectors, candidate_count)

    else:
        test_candidates = index.query(test["seq"].tolist(), candidate_count)

    ids, scores = aligned_neighbors(test["seq"].tolist(), references, test_candidates, best["mode"])
    probability = knn_probabilities(ids, scores, labels, best["k"])
    y_test = test["y_func"].to_numpy()
    test_result = {"retrieval": retrieval, "mode": best["mode"], "k": best["k"],
                   "candidates": candidate_count,
                   **metrics(y_test, probability, best["threshold"])}

    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(output / "validation_metrics.csv", index=False)
    pd.DataFrame([test_result]).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame({"name": test["name"].to_numpy(), "y_true": y_test,
                  "score": probability, "y_pred": (probability >= best["threshold"]).astype(int),
                  "nearest_name": train["name"].iloc[ids[:, 0]].to_numpy(),
                  "nearest_similarity": scores[:, 0]}).to_csv(output / "predictions.csv", index=False)
    report = pd.DataFrame([test_result])
    print("Лучшая конфигурация на validation:", best)
    print(report.to_string(index=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["pepanno", "amplify"], default="pepanno")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--mode", choices=["global", "local", "both"], default="both")
    parser.add_argument("--retrieval", choices=["esm2", "transformer", "dipeptide"], default="esm2")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    if args.candidates < 1 or args.batch_size < 1:
        parser.error("--candidates и --batch-size должны быть положительными")

    modes = ("global", "local") if args.mode == "both" else (args.mode,)
    train, val, test = load_frames(args.dataset)
    output = args.output or OUTPUT_ROOT / f"alignment_{args.retrieval}_{args.dataset}"
    checkpoint = args.checkpoint or OUTPUT_ROOT / ("simple" if args.dataset == "amplify"
                                                   else "pepanno") / "transformer.pt"
    run_experiment(train, val, test, output, args.candidates, modes,
                   args.retrieval, args.batch_size, args.device, checkpoint)


if __name__ == "__main__":
    main()
