import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors


def similarity_groups(sequences, threshold=0.6):
    """Связные компоненты пар с cosine similarity не ниже threshold."""
    if not 0 < threshold < 1:
        raise ValueError("threshold должен быть между 0 и 1")
    if len(sequences) < 2:
        raise ValueError("Нужны хотя бы две последовательности")

    vectors = TfidfVectorizer(analyzer="char", ngram_range=(3, 3),
                              lowercase=False).fit_transform(sequences)
    graph = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1).fit(
        vectors).radius_neighbors_graph(vectors, radius=1 - threshold,
                                        mode="connectivity")
    _, groups = connected_components(graph, directed=False)
    return groups


def grouped_train_val_split(frame, threshold=0.6, seed=42):
    """Выделить около 20% в validation без пересечения групп сходства."""
    groups = similarity_groups(frame["seq"].tolist(), threshold)
    labels = frame["y_func"].to_numpy()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    candidates = list(cv.split(np.zeros(len(frame)), labels, groups))
    train_ids, val_ids = min(
        candidates,
        key=lambda parts: (
            abs(len(parts[1]) / len(frame) - 0.2)
            + abs(labels[parts[1]].mean() - labels.mean()),
        ),
    )
    if not set(groups[train_ids]).isdisjoint(groups[val_ids]):
        raise RuntimeError("Группы сходства попали в обе части")
    if len(np.unique(labels[train_ids])) < 2 or len(np.unique(labels[val_ids])) < 2:
        raise ValueError("После разбиения в train/validation нужен каждый класс")
    return (frame.iloc[train_ids].reset_index(drop=True),
            frame.iloc[val_ids].reset_index(drop=True), groups)
