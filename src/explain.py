"""
Explain a single prediction: which features pushed the model toward
"phishing", in human-readable terms. This is the difference between an
analyst trusting an alert enough to act on it, and ignoring it as a
black box.

Usage:
    python src/explain.py --model models/model.joblib --url "http://paypa1-secure.tk/login"
"""

import argparse
import pandas as pd
import shap

from features import extract_features, FEATURE_NAMES
from model_io import load_model_bundle


def describe_factor(name, value):
    """Human-readable description of a feature, adapted to its actual value
    rather than a single fixed label -- a fixed label reads wrong when the
    value points the opposite direction (e.g. a large edit distance means
    the domain is NOT brand-like, not that it is). Kept in sync with
    app/streamlit_app.py's describe_factor()."""
    if name == "is_https":
        return "uses HTTPS" if value else "does not use HTTPS"
    if name == "brand_edit_distance":
        return (
            f"domain name is very close to a known brand name (edit distance {value})"
            if value <= 3
            else f"domain name is not close to any known brand name (edit distance {value})"
        )
    if name == "keyword_count":
        return (
            "contains no phishing-associated keywords" if value == 0
            else f"contains {value} phishing-associated keyword(s) (login, verify, secure, etc.)"
        )
    if name == "suspicious_tld":
        return "uses a top-level domain commonly abused for phishing" if value else "uses an ordinary top-level domain"
    if name == "num_hyphens":
        return f"contains {value} hyphen(s) in the URL"
    if name == "has_ip_host":
        return "uses a raw IP address instead of a domain name" if value else "uses a domain name, not a raw IP"
    if name == "brand_in_subdomain_not_domain":
        return (
            "a brand name appears in the subdomain but not the registered domain" if value
            else "no brand name mismatch between subdomain and registered domain"
        )
    if name == "domain_entropy":
        return f"domain character entropy is {value:.2f}"
    if name == "has_at_symbol":
        return "contains an '@' character, a known obfuscation technique" if value else "no '@' obfuscation present"
    return f"{name} = {value}"


def explain_url(model_path, url, top_k=5):
    clf, feature_names = load_model_bundle(model_path)

    feats = extract_features(url)
    X = pd.DataFrame([feats], columns=feature_names)

    prob = clf.predict_proba(X)[0, 1]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    # shap_values shape for binary RF: (n_samples, n_features, 2) or (n_samples, n_features)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0][:, 1] if shap_values.ndim == 3 else shap_values[0]

    contributions = pd.Series(sv, index=feature_names).sort_values(key=abs, ascending=False)

    print(f"URL: {url}")
    print(f"Phishing probability: {prob:.3f}  ->  {'FLAGGED' if prob >= 0.5 else 'not flagged'}\n")
    print(f"Top {top_k} contributing factors:")
    for name, val in contributions.head(top_k).items():
        direction = "raises" if val > 0 else "lowers"
        readable = describe_factor(name, feats[name])
        print(f"  - {readable}  ({direction} phishing score by {abs(val):.3f})")

    return prob, contributions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    explain_url(args.model, args.url)
