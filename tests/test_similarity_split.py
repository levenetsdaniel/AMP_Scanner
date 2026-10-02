import numpy as np
import pandas as pd

from src.grouped_validation import nearest_similarity
from src.similarity_split import grouped_train_val_split, similarity_groups


def test_grouped_split_keeps_similar_peptides_together():
    rng = np.random.default_rng(17)
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    sequences = []
    labels = []
    for i in range(30):
        original = "".join(rng.choice(alphabet, size=30))
        mutation = list(original)
        mutation[15] = "A" if mutation[15] != "A" else "C"
        sequences.extend([original, "".join(mutation)])
        labels.extend([i % 2, i % 2])
    frame = pd.DataFrame({"seq": sequences, "y_func": labels})

    groups = similarity_groups(frame.seq.tolist(), threshold=0.6)
    train, val, _ = grouped_train_val_split(frame, threshold=0.6)

    assert all(groups[2 * i] == groups[2 * i + 1] for i in range(30))
    assert set(train.seq).isdisjoint(val.seq)
    assert {0, 1}.issubset(train.y_func) and {0, 1}.issubset(val.y_func)
    assert nearest_similarity(train, val).max() < 0.6 + 1e-8
    for i in range(30):
        pair = set(sequences[2 * i:2 * i + 2])
        assert pair.issubset(set(train.seq)) or pair.issubset(set(val.seq))
