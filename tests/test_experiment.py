import pandas as pd
import pytest
import torch

from src.experiment import best_threshold, metrics
from src.features_extractor import seq2features
from src.peptide_dataset import PeptideDataset, collate_peptides
from src.peptide_transformer import PeptideTransformer


def test_threshold_and_metrics():
    y = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    threshold = best_threshold(y, probabilities)
    result = metrics(y, probabilities, threshold)
    assert result["roc_auc"] == 1
    assert result["f1"] == 1
    assert result["tn"] == 2 and result["tp"] == 2


def test_invalid_label_is_not_silently_cast_to_zero():
    df = pd.DataFrame({"seq": ["ACD"], "y_func": [0.5]})
    with pytest.raises(ValueError, match="метки 0 и 1"):
        PeptideDataset(df)


def test_features_reject_invalid_sequence():
    with pytest.raises(ValueError, match="непустая"):
        seq2features("")
    with pytest.raises(ValueError, match="20 стандартных"):
        seq2features("AXD")


def test_padding_does_not_change_prediction_and_empty_input_is_rejected():
    torch.manual_seed(7)
    model = PeptideTransformer().eval()
    single = torch.tensor([[1, 2, 3]])
    mixed = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]])
    with torch.inference_mode():
        expected = model(single, single.ne(0))
        actual = model(mixed, mixed.ne(0))[:1]
    torch.testing.assert_close(expected, actual, atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="хотя бы один остаток"):
        model(torch.zeros((1, 3), dtype=torch.long), torch.zeros((1, 3), dtype=torch.bool))


def test_collate_mask_and_gradient():
    frame = pd.DataFrame({"seq": ["ACD", "ACDEFG"], "y_func": [1, 0]})
    batch = collate_peptides([PeptideDataset(frame)[0], PeptideDataset(frame)[1]])
    assert batch["attention_mask"].sum(dim=1).tolist() == [3, 6]
    model = PeptideTransformer()
    loss = torch.nn.functional.cross_entropy(
        model(batch["input_ids"], batch["attention_mask"]), batch["labels"]
    )
    loss.backward()
    assert model.token_embedding.weight.grad[0].abs().sum() == 0
    assert model.token_embedding.weight.grad[1].abs().sum() > 0

