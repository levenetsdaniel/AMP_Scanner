"""Аудит происхождения и сдвига признаков между train/test PepAnno."""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features_extractor import ACIDS, extract_features
from .load_dataset import load_data


LENGTH_BINS = ((31, 50), (51, 70), (71, 100))
PAPER_URL = "https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1014369"


def name_family(name):
    if name.startswith("AMP_uniprot_"):
        return "train_uniprot_tag"
    if name.startswith("AMP_LAMP_"):
        return "train_LAMP_tag"
    if name.startswith("AMP_DRAMP"):
        return "train_DRAMP_tag"
    if name.startswith("AMP_AMPfun_"):
        return "train_AMPfun_tag"
    if re.match(r"^AMP_AMP\d+", name):
        return "train_AMP_tag"
    if re.match(r"^AMP_s\d+", name):
        return "train_s_tag"
    if name.startswith("AMP_"):
        return "train_other_AMP_tag"
    if name.startswith("sp|"):
        return "test_sp_format"
    if name.startswith("ALL"):
        return "test_ALL_format"
    return "other"


def add_descriptors(frame):
    result = frame.copy()
    features = extract_features(frame, "seq")
    result["length"] = frame.seq.str.len().to_numpy()
    for column in ("charge_density", "positive_frac", "negative_frac",
                   "hydrophobic_frac", "mean_hydropathy", "acid_C",
                   "acid_K", "acid_R", "acid_D", "acid_E"):
        result[column] = features[column].to_numpy()
    result["name_family"] = frame.name.map(name_family)
    return result, features


def exact_length_match(train, test, label, lower, upper, seed=42):
    left = train[(train.y_func == label) & train.length.between(lower, upper)]
    right = test[(test.y_func == label) & test.length.between(lower, upper)]
    rng = np.random.default_rng(seed)
    left_ids, right_ids = [], []
    for length in range(lower, upper + 1):
        a = left.index[left.length == length].to_numpy()
        b = right.index[right.length == length].to_numpy()
        count = min(len(a), len(b))
        if count:
            left_ids.extend(rng.choice(a, size=count, replace=False))
            right_ids.extend(rng.choice(b, size=count, replace=False))
    return np.asarray(left_ids, dtype=int), np.asarray(right_ids, dtype=int)


def domain_auc(train_features, test_features):
    x = pd.concat([train_features, test_features], ignore_index=True)
    y = np.r_[np.zeros(len(train_features), dtype=int),
              np.ones(len(test_features), dtype=int)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1,
                                                                max_iter=3000))
    scores = cross_val_score(model, x, y, scoring="roc_auc", cv=cv)
    return float(scores.mean()), float(scores.std(ddof=1))


def markdown_table(frame):
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def audit(output=Path("outputs/pepanno_data_audit"),
          report=Path("reports/pepanno_data_audit.md")):
    train, test = load_data("pepanno")
    train, train_features = add_descriptors(train)
    test, test_features = add_descriptors(test)
    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.concat([
        frame.groupby(["y_func", "y_type", "name_family"], dropna=False).size()
             .rename("n").reset_index().assign(split=split)
        for split, frame in (("train", train), ("test", test))
    ], ignore_index=True)
    metadata = metadata[["split", "y_func", "y_type", "name_family", "n"]]
    metadata.to_csv(output / "metadata.csv", index=False)

    columns = ["length", "charge_density", "positive_frac", "negative_frac",
               "hydrophobic_frac", "mean_hydropathy", "acid_C", "acid_K",
               "acid_R", "acid_D", "acid_E"]
    summaries = []
    shifts = []
    domains = []
    for label in (0, 1):
        for lower, upper in LENGTH_BINS:
            subset_name = f"{lower}-{upper}"
            left = train[(train.y_func == label) & train.length.between(lower, upper)]
            right = test[(test.y_func == label) & test.length.between(lower, upper)]
            for split, part in (("train", left), ("test", right)):
                row = {"split": split, "y_func": label, "length_bin": subset_name,
                       "n": len(part)}
                row.update({f"mean_{col}": float(part[col].mean()) for col in columns})
                summaries.append(row)

            for col in columns[1:]:
                pooled_sd = np.sqrt((left[col].var(ddof=1) + right[col].var(ddof=1)) / 2)
                difference = float(right[col].mean() - left[col].mean())
                shifts.append({"y_func": label, "length_bin": subset_name,
                               "feature": col, "test_minus_train": difference,
                               "standardized_difference": difference / pooled_sd
                               if pooled_sd else np.nan})

            left_ids, right_ids = exact_length_match(train, test, label, lower, upper)
            # Имена, метки AMP и длина не передаются классификатору.
            a = train_features.loc[left_ids, [f"acid_{acid}" for acid in ACIDS]]
            b = test_features.loc[right_ids, [f"acid_{acid}" for acid in ACIDS]]
            if min(len(a), len(b)) < 10:
                raise ValueError(f"Слишком мало пар длины для {label}, {subset_name}")
            mean_auc, sd_auc = domain_auc(a, b)
            domains.append({"y_func": label, "length_bin": subset_name,
                            "matched_per_split": len(a), "domain_auc": mean_auc,
                            "domain_auc_sd": sd_auc,
                            "matched_train_charge": float(train.loc[left_ids, "charge_density"].mean()),
                            "matched_test_charge": float(test.loc[right_ids, "charge_density"].mean())})

    summary = pd.DataFrame(summaries)
    shift = pd.DataFrame(shifts)
    domain = pd.DataFrame(domains)
    summary.to_csv(output / "strata.csv", index=False)
    shift.to_csv(output / "feature_shifts.csv", index=False)
    domain.to_csv(output / "domain_auc.csv", index=False)

    metadata_lines = markdown_table(metadata)
    domain_lines = markdown_table(domain.round(4))
    charge = summary[["split", "y_func", "length_bin", "n",
                      "mean_charge_density", "mean_positive_frac",
                      "mean_negative_frac", "mean_acid_C"]].round(4)
    charge_lines = markdown_table(charge)
    def charge_mean(split, length_bin):
        row = summary[(summary.split == split) & (summary.y_func == 1)
                      & (summary.length_bin == length_bin)]
        return float(row.iloc[0].mean_charge_density)

    report.write_text(
        "# Аудит train/test PepAnno AMP\n\n"
        "Запуск: `python -m src.pepanno_data_audit`. Анализируются очищенные CSV "
        "из `data/pepanno/`; исходный test используется **только для диагностики сдвига**. "
        "Ни одна модель AMP здесь не подбирается по test.\n\n"
        "## Метаданные\n\n"
        "`y_type` равен 0 для всех строк. Семейства `name` отражают только видимый "
        "шаблон идентификатора; они не доказывают происхождение каждой записи.\n\n"
        f"{metadata_lines}\n\n"
        "Имена train и test имеют непересекающиеся шаблоны. На test они также "
        "полностью разделяют классы: `sp|...` у класса 0, `ALL...` у класса 1. "
        "Использовать `name` как вход модели нельзя.\n\n"
        "## Состав внутри класса и диапазона длин\n\n"
        "Плотность заряда — прокси `(K+R-D-E)/L`, без pH, гистидина и концевых групп. "
        "Все доли и средние считаются по последовательностям, а не по остаткам.\n\n"
        f"{charge_lines}\n\n"
        "Полные различия признаков: "
        "[feature_shifts.csv](../outputs/pepanno_data_audit/feature_shifts.csv).\n\n"
        "## Отличимость train и test при одинаковой длине\n\n"
        "Для каждого класса и каждой **точной длины** берётся одинаковое число "
        "случайно выбранных train/test записей. Логистическая регрессия различает "
        "части только по 20 частотам аминокислот; результат — средний ROC-AUC "
        "пяти стратифицированных фолдов. 0.5 означает отсутствие различимого сигнала, "
        "большие значения — различимый сдвиг состава. Это диагностическая "
        "классификация происхождения, не качество AMP-модели.\n\n"
        f"{domain_lines}\n\n"
        "## Интерпретация и ограничения\n\n"
        "После точного сопоставления длин AMP train/test остаются различимы "
        "по аминокислотному составу заметно сильнее, чем не-AMP в диапазонах "
        f"51–100. У AMP 51–70 средняя плотность заряда падает с {charge_mean('train', '51-70'):.4f} "
        f"до {charge_mean('test', '51-70'):.4f}, у AMP 71–100 — с "
        f"{charge_mean('train', '71-100'):.4f} до {charge_mean('test', '71-100'):.4f}; "
        "у не-AMP этих длин она почти не "
        "меняется. Таким образом, проблема не сводится к распределению длин.\n\n"
        "Шаблоны ID и отличимость последовательностей согласуются с тем, что "
        "официальный test представляет отдельный источник. В статье PepAnno "
        "сказано, что AMP test заимствован из исследования Xu и что обучающие "
        "последовательности с ≥90% identity к test удалялись. "
        f"[Источник]({PAPER_URL}). Групповая validation внутри train не воспроизводит "
        "этот внешний способ формирования test. Удаление близких обучающих "
        "последовательностей делает снижение сходства ожидаемым, "
        "но само по себе не доказывает ошибку меток. Диагностика использует "
        "известный test и не должна становиться основанием для подбора новой модели "
        "на этом же test.\n",
        encoding="utf-8",
    )
    print("Метаданные:")
    print(metadata.to_string(index=False))
    print("\nДоменная классификация после точного сопоставления длин:")
    print(domain.round(4).to_string(index=False))
    print("\nОтчёт:", report)
    return metadata, summary, shift, domain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/pepanno_data_audit"))
    parser.add_argument("--report", type=Path, default=Path("reports/pepanno_data_audit.md"))
    args = parser.parse_args()
    audit(args.output, args.report)


if __name__ == "__main__":
    main()
