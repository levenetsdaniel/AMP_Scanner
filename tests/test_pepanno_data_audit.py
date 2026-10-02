import pandas as pd

from src.pepanno_data_audit import exact_length_match, name_family


def test_name_family_reports_format_without_using_label():
    assert name_family("AMP_LAMP_L01A001407_train") == "train_LAMP_tag"
    assert name_family("sp|Q91FT2|240R_IIV6") == "test_sp_format"
    assert name_family("ALL000004") == "test_ALL_format"


def test_exact_length_matching_balances_each_length():
    train = pd.DataFrame({"y_func": [1] * 6,
                          "length": [31, 31, 31, 32, 32, 33]})
    test = pd.DataFrame({"y_func": [1] * 6,
                         "length": [31, 31, 32, 32, 32, 34]})
    train_ids, test_ids = exact_length_match(train, test, 1, 31, 34)
    assert train_ids.tolist() == exact_length_match(train, test, 1, 31, 34)[0].tolist()
    assert train.loc[train_ids, "length"].value_counts().sort_index().to_dict() == {31: 2, 32: 2}
    assert test.loc[test_ids, "length"].value_counts().sort_index().to_dict() == {31: 2, 32: 2}
