"""
Train Logistic Regression, Random Forest, and XGBoost on the same data and
compare them on TWO things:
  1. Held-out accuracy (the easy, less meaningful number)
  2. Red-team catch rate (the number that actually matters for this project)

This is the comparison that should have been here from the start instead of
defaulting straight to Random Forest.

Usage:
    python src/compare_models.py --data ../data/train.csv
"""

import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb

from features import extract_features, FEATURE_NAMES
from adversarial import generate_adversarial_set
from model_io import save_model_bundle


def build_feature_matrix(urls):
    return pd.DataFrame([extract_features(u) for u in urls], columns=FEATURE_NAMES)


def get_models():
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        ),
    }


def red_team_catch_rate(model, n_per_technique=60):
    samples = generate_adversarial_set(n_per_technique)
    urls, techniques = zip(*samples)
    X_adv = build_feature_matrix(urls)
    probs = model.predict_proba(X_adv)[:, 1]
    flagged = (probs >= 0.5).astype(int)
    df = pd.DataFrame({"technique": techniques, "flagged": flagged})
    overall = flagged.mean()
    per_technique = df.groupby("technique")["flagged"].mean()
    return overall, per_technique


def main(data_path):
    df = pd.read_csv(data_path)
    X = build_feature_matrix(df["url"])
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    trained = {}

    for name, model in get_models().items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        held_out_f1 = f1_score(y_test, preds)
        held_out_auc = roc_auc_score(y_test, probs)
        catch_rate, per_technique = red_team_catch_rate(model)

        results.append({
            "model": name,
            "held_out_f1": round(held_out_f1, 4),
            "held_out_auc": round(held_out_auc, 4),
            "red_team_catch_rate": round(catch_rate, 4),
        })
        trained[name] = (model, per_technique)

        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=["benign", "phishing"]))
        print("Red-team catch rate by technique:")
        print(per_technique.round(3))

    summary = pd.DataFrame(results).sort_values("red_team_catch_rate", ascending=False)
    print("\n\n=== SUMMARY (sorted by what actually matters: red-team catch rate) ===")
    print(summary.to_string(index=False))

    best_name = summary.iloc[0]["model"]
    best_model, best_per_technique = trained[best_name]

    # Save the winning model as the single canonical model the app uses.
    save_model_bundle(best_model, FEATURE_NAMES, "../models/model.joblib")
    print(f"\nBest model on red-team catch rate: {best_name} -> saved to ../models/model.joblib")

    # Save the comparison itself as data, so the app can display *why* this
    # model was chosen without re-running training/red-teaming on every load.
    comparison_record = {
        "summary": summary.to_dict(orient="records"),
        "chosen_model": best_name,
        "chosen_model_per_technique": best_per_technique.round(4).to_dict(),
        "all_per_technique": {
            name: pt.round(4).to_dict() for name, (_, pt) in trained.items()
        },
    }
    with open("../data/model_comparison_results.json", "w") as f:
        json.dump(comparison_record, f, indent=2)
    print("Saved comparison results to ../data/model_comparison_results.json")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/train.csv")
    args = parser.parse_args()
    main(args.data)
