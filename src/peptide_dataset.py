import torch
from torch.utils.data import Dataset
import pandas as pd


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

VOCAB = {
    amino_acid: i + 1
    for i, amino_acid in enumerate(AMINO_ACIDS)
}

PAD_ID = 0

class PeptideDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.sequences = df["seq"].str.strip().str.upper().tolist()
        self.labels = df["y_func"].astype(int).tolist()

        for seq in self.sequences:
            if not seq or not set(seq).issubset(VOCAB):
                raise ValueError(f"Некорректная последовательность: {seq}")

        if not set(self.labels).issubset({0, 1}):
            raise ValueError("Ожидались метки 0 и 1")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        return {
            "input_ids": [
                VOCAB[amino_acid]
                for amino_acid in seq
            ],
            "label": self.labels[idx],
        }

def collate_peptides(batch):
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = torch.full(
        (len(batch), max_len),
        PAD_ID,
        dtype=torch.long,
    )

    labels = torch.tensor(
        [item["label"] for item in batch],
    )

    for i, item in enumerate(batch):
        ids = item["input_ids"]

        input_ids[i, :len(ids)] = torch.tensor(
            ids,
            dtype=torch.long,
        )

    attention_mask = input_ids != PAD_ID

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
