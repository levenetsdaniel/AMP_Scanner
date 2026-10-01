import numpy as np
import pandas as pd

ACIDS = "ACDEFGHIKLMNPQRSTVWY"
H = "AVILMFWY"
POSITIVE = "KR"
NEGATIVE = "DE"
TERMINUS_SIZE = 5
HELIX_ANGLE = np.deg2rad(100)
KD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5,
    "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3
}

FEATURE_NAMES = [f"acid_{a}" for a in ACIDS] + [
    "seq_len", "charge", "hydrophobic_frac", "mean_hydropathy",
    "positive_frac", "negative_frac", "charge_density",
    "n_terminal_positive_frac", "n_terminal_hydrophobic_frac",
    "c_terminal_positive_frac", "c_terminal_hydrophobic_frac",
    "hydrophobic_moment_helix",
]


def fraction_in(sequence: str, residues: str) -> float:
    return sum(a in residues for a in sequence) / len(sequence)


def seq2features(sequence: str) -> np.ndarray:
    if not isinstance(sequence, str):
        raise ValueError("Последовательность должна быть строкой")
    sequence = sequence.strip().upper()
    l = len(sequence)
    if l == 0 or not set(sequence).issubset(ACIDS):
        raise ValueError("Нужна непустая последовательность из 20 стандартных аминокислот")

    charge = sum(sequence.count(a) for a in POSITIVE) - sum(sequence.count(a) for a in NEGATIVE)
    hydropathy = np.array([KD[a] for a in sequence])
    positions = np.arange(l) * HELIX_ANGLE
    hydrophobic_moment = np.hypot(
        np.dot(hydropathy, np.cos(positions)),
        np.dot(hydropathy, np.sin(positions)),
    ) / l
    n_terminal = sequence[:TERMINUS_SIZE]
    c_terminal = sequence[-TERMINUS_SIZE:]

    features = [sequence.count(a) / l for a in ACIDS]
    features.extend([
        l,
        charge,
        fraction_in(sequence, H),
        float(hydropathy.mean()),
        fraction_in(sequence, POSITIVE),
        fraction_in(sequence, NEGATIVE),
        charge / l,
        fraction_in(n_terminal, POSITIVE),
        fraction_in(n_terminal, H),
        fraction_in(c_terminal, POSITIVE),
        fraction_in(c_terminal, H),
        hydrophobic_moment,
    ])
    return np.asarray(features, dtype=np.float32)


def extract_features(df: pd.DataFrame, column: str) -> pd.DataFrame:
    feature_series = df[column].apply(seq2features)
    feature_matrix = np.stack(feature_series.values)
    return pd.DataFrame(feature_matrix, columns=FEATURE_NAMES, index=df.index)
