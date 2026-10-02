import numpy as np
import pandas as pd
import pytest

from src import alignment_experiment
from src.alignment_experiment import (
    DipeptideCandidateIndex, EmbeddingCandidateIndex, aligned_neighbors,
    knn_probabilities, run_experiment,
)


def test_alignment_ranks_identical_peptide_first():
    references = ["ACDEFG", "KKKKKK", "VVVVVV"]
    ids = DipeptideCandidateIndex(references).query(["ACDEFG"], count=3)

    for mode in ("global", "local"):
        ranked_ids, scores = aligned_neighbors(["ACDEFG"], references, ids, mode)
        assert ranked_ids[0, 0] == 0
        assert scores[0, 0] == pytest.approx(1)
        assert np.all(np.diff(scores[0]) <= 0)


def test_knn_uses_alignment_scores_and_class_labels():
    ids = np.array([[2, 0, 1]])
    similarities = np.array([[0.9, 0.6, 0.3]])
    labels = np.array([0, 1, 1])

    assert knn_probabilities(ids, similarities, labels, 1)[0] == 1
    assert knn_probabilities(ids, similarities, labels, 3)[0] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="числа кандидатов"):
        knn_probabilities(ids, similarities, labels, 4)


def test_embedding_search_uses_cosine_similarity():
    index = EmbeddingCandidateIndex(np.array([[1.0, 0.0], [0.0, 1.0]]))
    ids = index.query(np.array([[0.9, 0.1]]), count=2)
    assert ids.tolist() == [[0, 1]]


def test_alignment_experiment_writes_separate_results(tmp_path):
    train = pd.DataFrame({"name": ["a", "b", "c", "d"],
                          "seq": ["AAAAAA", "CCCCCC", "GGGGGG", "KKKKKK"],
                          "y_func": [0, 0, 1, 1]})
    val = pd.DataFrame({"name": ["e", "f"], "seq": ["AAAAAA", "KKKKKK"],
                        "y_func": [0, 1]})
    test = pd.DataFrame({"name": ["g", "h"], "seq": ["CCCCCC", "GGGGGG"],
                         "y_func": [0, 1]})

    report = run_experiment(train, val, test, tmp_path, candidate_count=4,
                            modes=("global", "local"))

    assert report.loc[0, "roc_auc"] == 1
    assert report.loc[0, "retrieval"] == "dipeptide"
    assert (tmp_path / "validation_metrics.csv").exists()
    assert (tmp_path / "metrics.csv").exists()
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert predictions["nearest_name"].tolist() == ["b", "c"]


def test_esm_retrieval_uses_existing_embedding_function(tmp_path, monkeypatch):
    train = pd.DataFrame({"name": ["a", "b", "c", "d"],
                          "seq": ["AAAAAA", "CCCCCC", "GGGGGG", "KKKKKK"],
                          "y_func": [0, 0, 1, 1]})
    val = pd.DataFrame({"name": ["e", "f"], "seq": ["AAAAAA", "KKKKKK"],
                        "y_func": [0, 1]})
    test = pd.DataFrame({"name": ["g", "h"], "seq": ["CCCCCC", "GGGGGG"],
                         "y_func": [0, 1]})
    calls = []

    def fake_embeddings(model, tokenizer, sequences, batch_size, device, description):
        calls.append(description)
        return np.array([[seq.count(acid) for acid in "ACGK"] for seq in sequences])

    monkeypatch.setattr(alignment_experiment, "device_for", lambda name: "cpu")
    monkeypatch.setattr(alignment_experiment, "load_esm", lambda device: (object(), object()))
    monkeypatch.setattr(alignment_experiment, "esm_embeddings", fake_embeddings)

    report = run_experiment(train, val, test, tmp_path, candidate_count=4,
                            modes=("global",), retrieval="esm2")

    assert calls == ["ESM-2 reference", "ESM-2 validation", "ESM-2 test"]
    assert report.loc[0, "retrieval"] == "esm2"
    assert report.loc[0, "roc_auc"] == 1
