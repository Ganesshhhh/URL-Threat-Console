"""
Train the production model: XGBoost.

Why XGBoost and not something else: see compare_models.py and
data/model_comparison_results.json. Logistic Regression, Random Forest, and
XGBoost were all evaluated on both held-out accuracy and adversarial
red-team catch rate. XGBoost won on the metric that matters for this
project (red-team catch rate: 80.6% vs 78.6% for Random Forest and 65.3%
for Logistic Regression), while all three looked nearly identical on
held-out accuracy -- which is exactly why held-out accuracy alone is not
a trustworthy way to pick a model here.

Usage:
    python src/train.py --data data/train.csv --out models/model.joblib
"""

import argparse
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

from features import extract_features, FEATURE_NAMES
from model_io import save_model_bundle


def build_feature_matrix(urls):
    rows = [extract_features(u) for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def main(data_path, out_path):
    df = pd.read_csv(data_path)
    X = build_feature_matrix(df["url"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    print("=== Held-out test set performance ===")
    print(classification_report(y_test, preds, target_names=["benign", "phishing"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, probs):.4f}")

    importances = pd.Series(clf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
    print("\n=== Top 10 feature importances ===")
    print(importances.head(10))

    save_model_bundle(clf, FEATURE_NAMES, out_path)
    print(f"\nSaved model to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--out", default="models/model.joblib")
    args = parser.parse_args()
    main(args.data, args.out)
