from pathlib import Path
import pandas as pd
import requests


DATA_DIR = Path(__file__).resolve().parent / "data"

TRAIN_PATH = DATA_DIR / "pepanno_train.csv"
TEST_PATH = DATA_DIR / "pepanno_test.csv"

TRAIN_URL = "https://bis.zju.edu.cn/pepanno/download/pepdata/ampTrain"
TEST_URL = "https://bis.zju.edu.cn/pepanno/download/pepdata/ampTest"


def download_if_missing(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print(f"Используем локальный файл: {path}")
        return

    if not url.startswith(("https://", "http://")):
        raise ValueError(
            f"Укажи прямую ссылку для {path.name}"
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Скачиваем {path.name}...")

    temp_path = path.with_suffix(".csv.part")

    try:
        with requests.get(
            url,
            timeout=60,
            stream=True,
        ) as response:
            response.raise_for_status()

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        sample = pd.read_csv(temp_path, nrows=5)

        if sample.empty or len(sample.columns) < 2:
            raise ValueError(
                "Скачанный файл не похож на ожидаемый CSV"
            )

        temp_path.replace(path)

        print(f"Готово: {path}")

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

def compare_datasets(train_df, test_df):
    for dataset_name, df in [
        ("TRAIN", train_df),
        ("TEST", test_df),
    ]:
        print(f"\n{dataset_name}")

        for label in [0, 1]:
            lengths = df.loc[
                df["y_func"] == label, "seq"
            ].str.len()

            print(
                f"Class {label}: "
                f"n={len(lengths)}, "
                f"mean={lengths.mean():.1f}, "
                f"median={lengths.median():.1f}, "
                f"min={lengths.min()}, "
                f"max={lengths.max()}"
            )

def load_data():
    download_if_missing(TRAIN_URL, TRAIN_PATH)
    download_if_missing(TEST_URL, TEST_PATH)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train_without_collisions = train.groupby("seq").filter(lambda x: x['y_func'].nunique() == 1)
    test_without_collisions = test.groupby("seq").filter(lambda x: x['y_func'].nunique() == 1)

    train_final = train_without_collisions.drop_duplicates(subset=["seq", "y_func"], keep="first")
    test_final = test_without_collisions.drop_duplicates(subset=["seq", "y_func"], keep="first")

    print("Train:", train_final.shape)
    print("Test:", test_final.shape)

    compare_datasets(train_final, test_final)

    return train_final, test_final


if __name__ == "__main__":
    train, test = load_data()

    for name, df in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{name}")
        print("Shape:", df.shape)

        print("\ny_func:")
        print(df["y_func"].value_counts(dropna=False))

        print("\ny_type:")
        print(df["y_type"].value_counts(dropna=False))

        print("\nСовместное распределение:")
        print(pd.crosstab(df["y_func"], df["y_type"]))

        print("\nПримеры:")
        print(
            df[["name", "seq", "y_func", "y_type"]]
            .head(10)
            .to_string(index=False)
        )