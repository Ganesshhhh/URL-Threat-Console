"""
Red-team the model.

This is the point of the whole project: don't just report accuracy on a
held-out split from the SAME generative process the model was trained on
(that's an easy A+ that means nothing). Instead, generate URLs using
attack techniques the model was never explicitly trained to recognize as
a category, using logic independent from data/generate_synthetic_data.py,
and see where it actually breaks.

Techniques covered:
    1. Homograph substitution (visually similar unicode/ASCII lookalikes)
    2. Typosquatting (insertion, deletion, transposition, adjacent-key swap)
    3. Combosquatting (brand + unrelated word, on a clean TLD, HTTPS)
    4. URL shortener wrapping (simulated -- shortener hides all structure)
    5. Subdomain smuggling on a long legitimate-looking chain
    6. "Clean" phishing: HTTPS, short URL, no suspicious keywords, popular TLD
       (tests whether the model over-relies on superficial cues)

Usage:
    python src/adversarial.py --model models/model.joblib
"""

import argparse
import random
import string
import pandas as pd

from features import extract_features, FEATURE_NAMES, POPULAR_BRANDS
from model_io import load_model_bundle

random.seed(7)

QWERTY_ADJACENT = {
    "a": "qsz", "b": "vghn", "c": "xdfv", "d": "erfcxs", "e": "wsdr",
    "f": "rtgvcd", "g": "tyhbvf", "h": "yujnbg", "i": "ujko", "j": "uikmnh",
    "k": "ijolm", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "ol", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def _typo_insert(word):
    w = list(word)
    pos = random.randint(0, len(w))
    w.insert(pos, random.choice(string.ascii_lowercase))
    return "".join(w)


def _typo_delete(word):
    if len(word) < 2:
        return word
    pos = random.randint(0, len(word) - 1)
    return word[:pos] + word[pos + 1:]


def _typo_transpose(word):
    if len(word) < 2:
        return word
    pos = random.randint(0, len(word) - 2)
    w = list(word)
    w[pos], w[pos + 1] = w[pos + 1], w[pos]
    return "".join(w)


def _typo_adjacent_key(word):
    pos = random.randint(0, len(word) - 1)
    c = word[pos]
    repl = random.choice(QWERTY_ADJACENT.get(c, c))
    return word[:pos] + repl + word[pos + 1:]


def _homograph(word):
    swaps = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5", "g": "9"}
    chars = list(word)
    idxs = [i for i, c in enumerate(chars) if c in swaps]
    if not idxs:
        return word
    for i in random.sample(idxs, k=min(len(idxs), random.randint(1, 2))):
        chars[i] = swaps[chars[i]]
    return "".join(chars)


def generate_adversarial_set(n_per_technique=60):
    samples = []  # (url, technique)

    for _ in range(n_per_technique):
        brand = random.choice(POPULAR_BRANDS)
        fn = random.choice([_typo_insert, _typo_delete, _typo_transpose, _typo_adjacent_key])
        samples.append((f"https://{fn(brand)}.com/", "typosquat"))

    for _ in range(n_per_technique):
        brand = random.choice(POPULAR_BRANDS)
        samples.append((f"https://{_homograph(brand)}.com/account", "homograph"))

    for _ in range(n_per_technique):
        brand = random.choice(POPULAR_BRANDS)
        word = random.choice(["rewards", "support", "team", "portal", "id", "app", "cloud"])
        samples.append((f"https://{brand}{word}.com/", "combosquat_clean_tld"))

    for _ in range(n_per_technique):
        # simulated shortener output: no brand signal at all in the string
        code = "".join(random.choices(string.ascii_letters + string.digits, k=7))
        samples.append((f"https://bit.ly/{code}", "shortener"))

    for _ in range(n_per_technique):
        brand = random.choice(POPULAR_BRANDS)
        chain = ".".join(random.choices(["cdn", "static", "assets", "secure", "app", "my"], k=3))
        samples.append((f"https://{chain}.{brand}-verification.net/", "subdomain_smuggling"))

    for _ in range(n_per_technique):
        # "clean" phishing: HTTPS, short, popular TLD, no obvious keyword --
        # relies purely on the brand-adjacent name being unregistered by the brand
        brand = random.choice(POPULAR_BRANDS)
        word = random.choice(["hub", "now", "go", "pay", "wallet", "id"])
        samples.append((f"https://{brand}{word}.com/", "clean_phishing"))

    return samples


def evaluate(model_path, n_per_technique=60):
    clf, feature_names = load_model_bundle(model_path)

    samples = generate_adversarial_set(n_per_technique)
    urls, techniques = zip(*samples)
    X = pd.DataFrame([extract_features(u) for u in urls], columns=feature_names)
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df = pd.DataFrame({"url": urls, "technique": techniques, "phish_prob": probs, "flagged": preds})
    # ground truth: every sample here is an attack, so "flagged"==1 is a catch, ==0 is a miss
    summary = df.groupby("technique").agg(
        n=("flagged", "size"),
        caught=("flagged", "sum"),
    )
    summary["catch_rate"] = (summary["caught"] / summary["n"]).round(3)

    print("=== Adversarial red-team results (recall per attack technique) ===\n")
    print(summary)
    print(f"\nOverall catch rate: {df['flagged'].mean():.3f}")
    print("\nWorst 5 misses (lowest phishing probability assigned to an actual attack URL):")
    print(df.sort_values("phish_prob").head(5)[["url", "technique", "phish_prob"]].to_string(index=False))

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/model.joblib")
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()
    evaluate(args.model, args.n)
