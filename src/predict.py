"""Quick CLI: python src/predict.py --model models/model.joblib --url "http://..."."""
import argparse
import pandas as pd
from features import extract_features, FEATURE_NAMES
from model_io import load_model_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--url", required=True)
    args = parser.parse_args()

    clf, feature_names = load_model_bundle(args.model)
    X = pd.DataFrame([extract_features(args.url)], columns=feature_names)
    prob = clf.predict_proba(X)[0, 1]
    verdict = "PHISHING" if prob >= 0.5 else "benign"
    print(f"{args.url}\n  -> {verdict}  (phishing probability: {prob:.3f})")
