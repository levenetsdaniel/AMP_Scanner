import pandas as pd
import pytest

from src import load_dataset


class FakeResponse:
    def __init__(self, content):
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.content[:10]
        yield self.content[10:]


def test_missing_pepanno_file_is_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(load_dataset, "PEPANNO_DIR", tmp_path)
    content = b"name,seq,y_func,y_type\na,ACD,1,0\n"
    called = []

    def fake_get(url, timeout, stream):
        called.append((url, timeout, stream))
        return FakeResponse(content)

    monkeypatch.setattr(load_dataset.requests, "get", fake_get)
    path = load_dataset.download_pepanno_file("pepanno_train.csv")
    assert path.read_bytes() == content
    assert called == [(load_dataset.PEPANNO_URLS["pepanno_train.csv"], 60, True)]
    assert load_dataset.download_pepanno_file("pepanno_train.csv") == path
    assert len(called) == 1


def test_bad_download_is_not_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(load_dataset, "PEPANNO_DIR", tmp_path)
    monkeypatch.setattr(load_dataset.requests, "get",
                        lambda url, timeout, stream: FakeResponse(b"<html>error</html>"))
    with pytest.raises(ValueError, match="не похож на CSV"):
        load_dataset.download_pepanno_file("pepanno_test.csv")
    assert not (tmp_path / "pepanno_test.csv").exists()
    assert not (tmp_path / "pepanno_test.csv.part").exists()


def test_pepanno_conflicting_labels_are_removed(tmp_path, monkeypatch):
    monkeypatch.setattr(load_dataset, "PEPANNO_DIR", tmp_path)
    train = pd.DataFrame({"name": ["a", "b", "c", "d"],
                          "seq": ["ACD", "ACD", "EFG", "HIK"],
                          "y_func": [0, 1, 0, 1], "y_type": [0] * 4})
    test = pd.DataFrame({"name": ["e", "f"], "seq": ["LMN", "PQR"],
                         "y_func": [0, 1], "y_type": [0, 0]})
    train.to_csv(tmp_path / "pepanno_train.csv", index=False)
    test.to_csv(tmp_path / "pepanno_test.csv", index=False)
    cleaned_train, cleaned_test = load_dataset.load_pepanno()
    assert set(cleaned_train.seq) == {"EFG", "HIK"}
    assert len(cleaned_test) == 2
