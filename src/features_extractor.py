import numpy as np
import pandas as pd

ACIDS = "ACDEFGHIKLMNPQRSTVWY"
H = "AVILMFWY"
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5,
    "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}

def seq2features(sequence: str) -> np.ndarray:
    l = len(sequence)
    if l == 0:
        return np.zeros(24)

    features = [sequence.count(a) / l for a in ACIDS]  # 20 признаков

    features.append(l)

    charge_index = sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")
    features.append(charge_index)

    hydrophobic_fraction = sum(a in H for a in sequence) / l
    features.append(hydrophobic_fraction)

    mean_hydropathy = sum(KD.get(a, 0.0) for a in sequence) / l
    features.append(mean_hydropathy)

    features_array = np.array(features, dtype=np.float32)

    assert features_array.shape == (24,), f"Ожидалось 24 признака, получено {features_array.shape[0]}"
    return features_array

def extract_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    features_df = df.copy()
    feature_series = features_df[column].apply(seq2features)
    feature_matrix = np.stack(feature_series.values)
    col_names = [f"acid_{a}" for a in ACIDS] + ["seq_len", "charge", "hydrophobic_frac", "mean_hydropathy"]
    features_df = pd.DataFrame(feature_matrix, columns=col_names, index=df.index)
    return features_df