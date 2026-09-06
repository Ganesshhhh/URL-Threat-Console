"""
OPTIONAL augmentation step: mix in a small number of harder phishing/benign
examples on top of the real PhiUSIIL data (see load_phiusiil.py), if you
find the model trained on real data alone still has a specific blind spot
in the adversarial red-team results (src/adversarial.py).

This is not a substitute for real training data -- run
data/load_phiusiil.py first. This script only appends to whatever
data/train.csv already contains.
"""

import argparse
import random
import string
import csv
from pathlib import Path

random.seed(99)

POPULAR_BRANDS = [
    "paypal", "google", "amazon", "microsoft", "apple", "netflix", "facebook",
    "instagram", "chase", "bankofamerica", "wellsfargo", "coinbase", "binance",
    "metamask", "dropbox", "linkedin", "twitter", "adobe", "outlook", "steam",
]

WORDS = ["pay", "hub", "id", "now", "go", "app", "portal", "rewards", "team", "cloud", "wallet"]


def typo_insert(word):
    w = list(word)
    w.insert(random.randint(0, len(w)), random.choice(string.ascii_lowercase))
    return "".join(w)


def homograph(word):
    swaps = {"o": "0", "l": "1", "i": "1", "e": "3", "a": "4", "s": "5"}
    chars = list(word)
    idxs = [i for i, c in enumerate(chars) if c in swaps]
    if idxs:
        i = random.choice(idxs)
        chars[i] = swaps[chars[i]]
    return "".join(chars)


def make_hard_phish():
    brand = random.choice(POPULAR_BRANDS)
    kind = random.random()
    if kind < 0.3:
        # combosquat, clean TLD, https, no keyword
        return f"https://{brand}{random.choice(WORDS)}.com/"
    elif kind < 0.6:
        # typosquat, https, clean TLD
        return f"https://{typo_insert(brand)}.com/"
    elif kind < 0.8:
        # homograph, https, clean TLD
        return f"https://{homograph(brand)}.com/account"
    else:
        # subdomain smuggling with clean-looking chain
        chain = ".".join(random.choices(["cdn", "static", "my", "app"], k=2))
        return f"https://{chain}.{brand}-support.net/"


def make_hard_benign():
    # legitimate-looking small brand/startup domains that just happen to
    # contain common words, so the model doesn't learn "any word == phishing"
    word = random.choice(WORDS) + random.choice(["ly", "io", "co", "app", "hq"])
    return f"https://{word}.com/"


def augment(train_csv="data/train.csv", n=800, force=False):
    train_path = Path(train_csv)

    # This script only APPENDS, so re-running it (or invoking it with an
    # unrelated flag like --help by mistake) would silently duplicate the
    # augmented rows. Guard with a sentinel marker file next to the CSV.
    marker_path = train_path.with_suffix(train_path.suffix + ".hard_examples_added")
    if train_path.exists() and marker_path.exists() and not force:
        raise SystemExit(
            f"{train_csv} already has hard examples appended (see "
            f"{marker_path}). Re-running would duplicate them. "
            f"Pass --force to append anyway, or --train_csv to target "
            f"a different file."
        )

    rows = []
    for _ in range(n // 2):
        rows.append((make_hard_phish(), 1))
    for _ in range(n // 2):
        rows.append((make_hard_benign(), 0))
    random.shuffle(rows)

    with open(train_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    marker_path.touch()

    print(f"Appended {len(rows)} hard examples to {train_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OPTIONAL: append synthetic hard examples to an existing train.csv."
    )
    parser.add_argument("--train_csv", default="data/train.csv",
                         help="CSV to append to (must already exist with a url,label header).")
    parser.add_argument("--n", type=int, default=800,
                         help="Total number of hard examples to generate (split evenly phish/benign).")
    parser.add_argument("--force", action="store_true",
                         help="Append even if this file already had hard examples added before.")
    args = parser.parse_args()
    augment(train_csv=args.train_csv, n=args.n, force=args.force)
