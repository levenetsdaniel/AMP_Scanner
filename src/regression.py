import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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



def prepare_data(df: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = extract_features(df.drop(columns="y_func"), column="seq").values
    y = df["y_func"]
    return train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

def train_linear_model(model, X_train: pd.DataFrame, y_train: pd.Series):
    model.fit(X_train, y_train)

def evaluate_linear_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))
    print("PR-AUC:", average_precision_score(y_test, y_prob))
    print("F1:", f1_score(y_test, y_pred))
    print("MCC:", matthews_corrcoef(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

def main(seed: int = 42, max_iter: int = 1000) -> None:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=max_iter,
            random_state=seed
        )
    )

    train_df, test_df = load_data()

    X_train = extract_features(train_df, column="seq").values

    y_train = train_df["y_func"]

    X_test = extract_features(test_df, column="seq").values

    y_test = test_df["y_func"]

    print("\nTrain extracted features:", X_train.shape)
    print("Test extracted features:", X_test.shape)

    train_linear_model(model, X_train, y_train)

    evaluate_linear_model(model, X_test, y_test)

if __name__ == "__main__":
    main()

