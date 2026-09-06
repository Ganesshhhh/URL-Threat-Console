"""
Load the PhiUSIIL Phishing URL Dataset and produce data/train.csv in the
schema the rest of this project expects: two columns, url and label,
where label=1 means phishing and label=0 means benign.

PhiUSIIL itself uses the OPPOSITE convention (label=1 is legitimate,
label=0 is phishing) -- this loader flips it. Getting this backwards
silently trains a model that's inverted, so the flip is asserted, not
just assumed (see _sanity_check_labels below).

Source: Prasad & Chandra, "PhiUSIIL: A diverse security profile empowered
phishing URL detection framework based on similarity index and
incremental learning", Computers & Security, 2024.
134,850 legitimate + 100,945 phishing URLs.
UCI: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset
Kaggle mirror: https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset

THREE WAYS TO GET THE DATA IN (tried in this order):

1. --raw_csv path/to/file.csv
   Point directly at a CSV you already downloaded (e.g. from Kaggle).

2. data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
   Drop the downloaded CSV here with this exact name and this script finds
   it automatically -- no flag needed.

3. Automatic fetch via the `ucimlrepo` package (pip install ucimlrepo).
   This is the easiest path with no Kaggle account/API key needed, but it
   makes a live network call to archive.ics.uci.edu, so it only works on a
   machine with normal internet access.

Usage:
    python data/load_phiusiil.py --out data/train.csv
    python data/load_phiusiil.py --raw_csv ~/Downloads/PhiUSIIL.csv --out data/train.csv
    python data/load_phiusiil.py --max_rows 20000   # faster iteration
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

DEFAULT_RAW_PATH = Path(__file__).parent / "raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"


def _find_column(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _load_from_csv(path):
    print(f"Loading PhiUSIIL from local CSV: {path}")
    return pd.read_csv(path)


def _load_from_ucimlrepo():
    print("No local CSV found -- attempting live fetch via ucimlrepo "
          "(archive.ics.uci.edu). This requires internet access.")
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as e:
        raise SystemExit(
            "ucimlrepo is not installed. Run: pip install ucimlrepo\n"
            "Or download the CSV manually and pass it with --raw_csv "
            "(see this file's docstring for links)."
        ) from e

    dataset = fetch_ucirepo(id=967)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    return df


def _sanity_check_labels(df, url_col, label_col):
    """
    PhiUSIIL's own documentation states label=1 is legitimate, label=0 is
    phishing. Confirm that assumption against the data itself rather than
    trusting it blindly: legitimate URLs in this dataset are disproportionately
    HTTPS, so if label==1 rows are NOT mostly https, the polarity assumption
    is wrong and label needs to NOT be flipped.
    """
    https_rate_label1 = df[df[label_col] == 1][url_col].str.startswith("https").mean()
    https_rate_label0 = df[df[label_col] == 0][url_col].str.startswith("https").mean()
    print(f"HTTPS rate where label==1: {https_rate_label1:.2%}")
    print(f"HTTPS rate where label==0: {https_rate_label0:.2%}")
    if https_rate_label1 < https_rate_label0:
        print("WARNING: label==1 has a LOWER https rate than label==0. This "
              "contradicts the documented convention (1=legitimate). Check "
              "the source file -- the flip below may be backwards.")
    return https_rate_label1 >= https_rate_label0


def main(raw_csv, out_path, max_rows):
    if raw_csv:
        df = _load_from_csv(raw_csv)
    elif DEFAULT_RAW_PATH.exists():
        df = _load_from_csv(DEFAULT_RAW_PATH)
    else:
        df = _load_from_ucimlrepo()

    url_col = _find_column(df, ["URL", "url"])
    label_col = _find_column(df, ["label", "Label", "CLASS_LABEL", "class"])

    if url_col is None or label_col is None:
        print(f"Columns found: {list(df.columns)}", file=sys.stderr)
        raise SystemExit(
            "Could not find a URL column and/or a label column by name. "
            "Open the CSV, check the actual column names, and adjust "
            "_find_column's candidate lists above."
        )

    polarity_confirmed = _sanity_check_labels(df, url_col, label_col)
    df = df[[url_col, label_col]].rename(columns={url_col: "url", label_col: "_raw_label"})

    # Flip to this project's convention: 1 = phishing, 0 = benign.
    if polarity_confirmed:
        df["label"] = 1 - df["_raw_label"]
    else:
        df["label"] = df["_raw_label"]  # already matches, per the sanity check
    df = df.drop(columns=["_raw_label"])

    df = df.dropna(subset=["url"])
    df["label"] = df["label"].astype(int)

    if max_rows and len(df) > max_rows:
        df = (
            df.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max_rows // 2), random_state=42))
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    print(f"\nFinal dataset: {len(df)} rows")
    print(df["label"].value_counts().rename({0: "benign", 1: "phishing"}))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv", default=None, help="Path to a downloaded PhiUSIIL CSV")
    parser.add_argument("--out", default="data/train.csv")
    parser.add_argument("--max_rows", type=int, default=20000,
                         help="Cap total rows for faster iteration. 0 = use all ~235k rows.")
    args = parser.parse_args()
    main(args.raw_csv, args.out, args.max_rows)
