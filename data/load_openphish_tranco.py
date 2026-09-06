"""
Load phishing URLs from OpenPhish (using pyopdb database / OpenPhish feed)
and benign URLs from Tranco crawler/ranking list to produce data/train.csv.

Schema:
- url: string
- label: int (1 = phishing, 0 = benign)

Usage:
    python data/load_openphish_tranco.py --out data/train.csv --max_rows 20000
"""

import argparse
import os
import sys
import sqlite3
import random
from pathlib import Path

import pandas as pd
import requests

# Try importing tranco
try:
    from tranco import Tranco
    TRANCO_AVAILABLE = True
except ImportError:
    TRANCO_AVAILABLE = False

# Path to pyopdb module directory
PYOPDB_DIR = Path(__file__).parent / "pyopdb"
if PYOPDB_DIR.exists():
    sys.path.insert(0, str(PYOPDB_DIR))

try:
    from pyopdb import OPDB
    PYOPDB_AVAILABLE = True
except ImportError:
    PYOPDB_AVAILABLE = False


def fetch_openphish_urls():
    """
    Fetch phishing URLs from OpenPhish via pyopdb SQLite databases and
    from the live OpenPhish feed (https://openphish.com/feed.txt).
    """
    urls = set()

    # 1. Try pyopdb SQLite databases (sample db or custom db)
    db_paths = [
        PYOPDB_DIR / "opdb-sample.db",
        PYOPDB_DIR / "opdb.db",
        Path("opdb.db"),
        Path("data/opdb.db"),
    ]

    for db_path in db_paths:
        if db_path.exists():
            print(f"Loading phishing URLs from pyopdb SQLite database: {db_path}")
            try:
                conn = sqlite3.connect(str(db_path))
                cursor = conn.cursor()
                cursor.execute("SELECT DISTINCT url FROM phishing_urls")
                rows = cursor.fetchall()
                count_db = 0
                for r in rows:
                    if r[0] and isinstance(r[0], str) and r[0].startswith(("http://", "https://")):
                        urls.add(r[0].strip())
                        count_db += 1
                conn.close()
                print(f"Loaded {count_db} URLs from {db_path}.")
            except Exception as e:
                print(f"Note: Could not query {db_path}: {e}")

    # 2. Fetch live feed from OpenPhish (openphish.com/feed.txt)
    print("Fetching live phishing URLs from OpenPhish feed (https://openphish.com/feed.txt)...")
    try:
        resp = requests.get("https://openphish.com/feed.txt", timeout=10)
        if resp.status_code == 200:
            lines = resp.text.splitlines()
            live_urls = [line.strip() for line in lines if line.strip().startswith(("http://", "https://"))]
            urls.update(live_urls)
            print(f"Fetched {len(live_urls)} live phishing URLs from OpenPhish.")
        else:
            print(f"OpenPhish feed returned status code {resp.status_code}")
    except Exception as e:
        print(f"Warning: Could not fetch live OpenPhish feed: {e}")

    phishing_list = list(urls)
    print(f"Total unique OpenPhish phishing URLs collected: {len(phishing_list)}")
    return phishing_list


def fetch_tranco_urls(count=10000):
    """
    Fetch top domains using Tranco crawler library and transform them into benign URLs.
    """
    if not TRANCO_AVAILABLE:
        raise RuntimeError("The 'tranco' package is not installed. Run: pip install tranco")

    print(f"Fetching top {count} domains from Tranco ranking list...")
    t = Tranco(cache=True, cache_dir=".tranco_cache")
    tranco_list = t.list()
    top_domains = tranco_list.top(count)

    print(f"Retrieved {len(top_domains)} domains from Tranco.")

    benign_urls = []
    # Common paths for realistic URL structures
    sample_paths = [
        "", "/", "/index.html", "/about", "/contact", "/home", "/products",
        "/services", "/faq", "/login", "/search", "/en/", "/blog", "/news",
        # Multi-segment paths (repositories, profiles, articles, documentation)
        "/openphish/pyopdb", "/facebook/react", "/torvalds/linux", "/python/cpython",
        "/microsoft/vscode", "/google/guava", "/org/repository-name", "/user/project",
        "/docs/v2/getting-started", "/blog/2026/01/introducing-new-features",
        "/wiki/Main_Page", "/wiki/Special:Search", "/r/technology/comments/general_discussion",
        "/articles/security-best-practices", "/item/detail/10293847", "/v1/api/health",
        "/en-us/download/installer.html", "/help/center/article/40928"
    ]

    query_params = [
        "", "", "", "",
        "?ref=home", "?q=search+query", "?id=10923", "?lang=en&theme=dark",
        "?page=2&sort=latest", "?utm_source=google"
    ]

    random.seed(42)
    subdomain_choices = ["", "", "www", "www", "www", "m", "blog", "news", "app"]
    for domain in top_domains:
        scheme = "https" if random.random() < 0.85 else "http"
        sub = random.choice(subdomain_choices)
        host = f"{sub}.{domain}" if sub else domain
        path = random.choice(sample_paths)
        qp = random.choice(query_params)
        url = f"{scheme}://{host}{path}{qp}"
        benign_urls.append(url)

    return benign_urls


def main(out_path, max_rows):
    phishing_urls = fetch_openphish_urls()
    if not phishing_urls:
        raise RuntimeError("No phishing URLs were collected from OpenPhish/pyopdb.")

    target_per_class = max_rows // 2 if max_rows > 0 else len(phishing_urls)
    num_phishing = min(len(phishing_urls), target_per_class) if max_rows > 0 else len(phishing_urls)
    
    # Randomly sample phishing URLs if we have more than target_per_class
    random.seed(42)
    if len(phishing_urls) > num_phishing:
        phishing_sampled = random.sample(phishing_urls, num_phishing)
    else:
        phishing_sampled = phishing_urls

    num_benign = len(phishing_sampled) if max_rows > 0 else max(10000, len(phishing_sampled))
    benign_urls = fetch_tranco_urls(count=num_benign)

    df_phishing = pd.DataFrame({"url": phishing_sampled, "label": 1})
    df_benign = pd.DataFrame({"url": benign_urls, "label": 0})

    df = pd.concat([df_phishing, df_benign], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\nFinal dataset composition:")
    print(df["label"].value_counts().rename({0: "benign (Tranco)", 1: "phishing (OpenPhish)"}))
    print(f"Total rows: {len(df)}")

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"\nWrote dataset to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build training dataset from OpenPhish (pyopdb) and Tranco.")
    parser.add_argument("--out", default="data/train.csv", help="Output CSV path")
    parser.add_argument("--max_rows", type=int, default=20000,
                        help="Cap total rows for training (0 = use all available)")
    args = parser.parse_args()
    main(args.out, args.max_rows)
