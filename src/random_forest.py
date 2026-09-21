import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from load_dataset import load_data
from features_extractor import extract_features
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)

def train_random_forest(model, X_train: pd.DataFrame, y_train: pd.Series):
    model.fit(X_train, y_train)

def evaluate_random_forest(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC:", average_precision_score(y_test, y_prob))
    print("F1:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def main(seed: int = 42, n_estimators: int = 100) -> None:
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed)

    train_df, test_df = load_data()

    X_train = extract_features(train_df, column="seq").values

    y_train = train_df["y_func"]

    X_test = extract_features(test_df, column="seq").values

    y_test = test_df["y_func"]

    print("\nTrain extracted features:", X_train.shape)
    print("Test extracted features:", X_test.shape)

    train_random_forest(model, X_train, y_train)

    evaluate_random_forest(model, X_test, y_test)

if __name__ == "__main__":
    main()