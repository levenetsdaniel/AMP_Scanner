# load_dataset.py

from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent / "data"

AMPLIFY_DIR = DATA_DIR / "amplify"
PEPANNO_DIR = DATA_DIR / "pepanno"

BASE_URL = (
    "https://raw.githubusercontent.com/"
    "BirolLab/AMPlify/master/data"
)

AMPLIFY_FILES = {
    "train_positive": "AMPlify_AMP_train_common.fa",
    "train_negative": "AMPlify_non_AMP_train_balanced.fa",
    "test_positive": "AMPlify_AMP_test_common.fa",
    "test_negative": "AMPlify_non_AMP_test_balanced.fa",
}

PEPANNO_URLS = {
    "pepanno_train.csv": "https://bis.zju.edu.cn/pepanno/download/pepdata/ampTrain",
    "pepanno_test.csv": "https://bis.zju.edu.cn/pepanno/download/pepdata/ampTest",
}
PEPANNO_COLUMNS = {"name", "seq", "y_func", "y_type"}

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def download_file(filename: str) -> Path:
    AMPLIFY_DIR.mkdir(parents=True, exist_ok=True)

    destination = AMPLIFY_DIR / filename

    if destination.exists() and destination.stat().st_size > 0:
        print("Используем локальный файл:", destination)
        return destination

    url = f"{BASE_URL}/{filename}"

    print("Скачиваем:", url)

    response = requests.get(url, timeout=60)
    response.raise_for_status()

    content = response.text

    if not content.lstrip().startswith(">"):
        raise ValueError(f"Файл {filename} не похож на FASTA.")

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)

    return destination


def read_fasta(path: Path, label: int) -> pd.DataFrame:
    records = []

    name = None
    sequence_parts = []

    def append_record():
        if name is None:
            return

        sequence = "".join(sequence_parts).strip().upper()

        records.append(
            {
                "name": name,
                "seq": sequence,
                "y_func": label,
                "y_type": 0,
            }
        )

    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith(">"):
                append_record()

                name = line[1:].strip()
                sequence_parts = []
            else:
                if name is None:
                    raise ValueError(f"Некорректный FASTA: {path}")

                sequence_parts.append(line)

    append_record()

    return pd.DataFrame(
        records,
        columns=["name", "seq", "y_func", "y_type"],
    )


def validate_dataset(train: pd.DataFrame, test: pd.DataFrame) -> None:
    for split_name, frame in (("train", train), ("test", test)):
        if frame.empty or frame["seq"].isna().any():
            raise ValueError(f"{split_name}: пустые данные или последовательности")

        if not frame["y_func"].isin([0, 1]).all():
            raise ValueError(f"{split_name}: допустимы только метки 0 и 1")

        if not frame["seq"].map(lambda seq: isinstance(seq, str) and bool(seq) and set(seq).issubset(AMINO_ACIDS)).all():
            raise ValueError(f"{split_name}: некорректная аминокислотная последовательность")

        if (frame.groupby("seq")["y_func"].nunique() > 1).any():
            raise ValueError(f"{split_name}: конфликтующие метки одной последовательности")

    if set(train["seq"]) & set(test["seq"]):
        raise ValueError("Точные совпадения между train и test")


def print_statistics(train: pd.DataFrame, test: pd.DataFrame):
    print("Train:", train.shape)
    print("Test:", test.shape)

    for split_name, df in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{split_name}")

        for label in [0, 1]:
            lengths = df.loc[
                df["y_func"] == label, "seq"
            ].str.len()

            if lengths.empty:
                raise ValueError(
                    f"{split_name}: отсутствует класс {label}"
                )

            print(
                f"Class {label}: "
                f"n={len(lengths)}, "
                f"mean={lengths.mean():.1f}, "
                f"median={lengths.median():.1f}, "
                f"min={lengths.min()}, "
                f"max={lengths.max()}"
            )


def load_amplify():
    paths = {
        key: download_file(filename)
        for key, filename in AMPLIFY_FILES.items()
    }

    train = pd.concat(
        [
            read_fasta(paths["train_positive"], label=1),
            read_fasta(paths["train_negative"], label=0),
        ],
        ignore_index=True,
    )

    test = pd.concat(
        [
            read_fasta(paths["test_positive"], label=1),
            read_fasta(paths["test_negative"], label=0),
        ],
        ignore_index=True,
    )

    validate_dataset(train, test)
    print_statistics(train, test)

    return train, test


def download_pepanno_file(filename: str) -> Path:
    destination = PEPANNO_DIR / filename
    if destination.exists() and destination.stat().st_size > 0:
        print("Используем локальный файл:", destination)
        return destination

    PEPANNO_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print("Скачиваем:", PEPANNO_URLS[filename])

    try:
        with requests.get(PEPANNO_URLS[filename], timeout=60, stream=True) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)

        sample = pd.read_csv(temporary, nrows=5)
        if sample.empty or not PEPANNO_COLUMNS.issubset(sample.columns):
            raise ValueError(f"Скачанный {filename} не похож на CSV PepAnno")

        temporary.replace(destination)

    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return destination


def load_pepanno():
    train = pd.read_csv(download_pepanno_file("pepanno_train.csv"))
    test = pd.read_csv(download_pepanno_file("pepanno_test.csv"))

    for split_name, frame in (("train", train), ("test", test)):
        if not PEPANNO_COLUMNS.issubset(frame.columns):
            raise ValueError(f"PepAnno {split_name}: нужны колонки {sorted(PEPANNO_COLUMNS)}")

        frame["seq"] = frame["seq"].str.strip().str.upper()

    conflicts = train.groupby("seq")["y_func"].transform("nunique") > 1
    removed = int(conflicts.sum())
    train = train.loc[~conflicts].drop_duplicates(subset="seq").reset_index(drop=True)
    test = test.drop_duplicates(subset="seq").reset_index(drop=True)
    print(f"PepAnno: исключено {removed} строк с конфликтующими метками")

    validate_dataset(train, test)
    print_statistics(train, test)
    return train, test


def load_data(dataset: str = "amplify"):
    dataset = dataset.strip().lower()

    if dataset == "amplify":
        return load_amplify()

    if dataset == "pepanno":
        return load_pepanno()

    raise ValueError(f"Неизвестный dataset={dataset!r}; выбери amplify или pepanno")


if __name__ == "__main__":
    load_data()
