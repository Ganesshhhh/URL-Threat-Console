"""
Feed worker for the live-monitoring tab.

IMPORTANT DESIGN NOTE: this does NOT retrain the model on the incoming
stream. A Certificate Transparency log gives you a domain name the moment
a certificate is issued for it -- it does not give you a label saying
"this turned out to be phishing." Retraining a supervised model on
unlabeled data isn't meaningful. What this worker actually automates is
the SCORING step: it takes the already-trained model, applies it to every
new domain as it appears, and appends anything that clears the threshold
to a feed file. The site's monitoring tab reads that feed and updates
without you doing anything -- new flags show up on their own -- but the
model's weights are fixed at training time.

Two modes:
  - run_live(): connects to the real Certificate Transparency stream via
    certstream. Needs outbound internet access.
  - run_simulated(): generates a synthetic stream of newly-registered-looking
    domains (mix of benign and attack patterns) at a fixed interval, so the
    monitoring UI can be exercised and demoed without a network dependency.

Both write JSON lines to the same feed file, one line per scored domain
that was flagged, so the UI code doesn't need to know which mode produced
them.
"""

import argparse
import json
import random
import string
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from features import extract_features, FEATURE_NAMES, POPULAR_BRANDS, closest_brand_distance
from model_io import load_model_bundle

try:
    import certstream
except ImportError:
    certstream = None


def _load_model(model_path):
    return load_model_bundle(model_path)


def _score_domain(clf, feature_names, domain):
    url = f"https://{domain}/"
    feats = extract_features(url)
    X = pd.DataFrame([feats], columns=feature_names)
    prob = float(clf.predict_proba(X)[0, 1])
    return prob, feats


def _append_to_feed(feed_path, record):
    with open(feed_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def _is_brand_relevant(domain, max_edit_distance=3):
    base = domain.split(".")[0]
    dist, brand = closest_brand_distance(base)
    return dist <= max_edit_distance or any(b in domain.lower() for b in POPULAR_BRANDS)


# ---------------------------------------------------------------------------
# Simulated mode (no network required)
# ---------------------------------------------------------------------------

def _random_domain(attack_like: bool):
    brand = random.choice(POPULAR_BRANDS)
    if not attack_like:
        word = random.choice(["hq", "labs", "dev", "media"]) + "".join(
            random.choices(string.ascii_lowercase, k=4)
        )
        return f"{word}.com"
    kind = random.random()
    if kind < 0.4:
        word = random.choice(["pay", "id", "hub", "support", "portal"])
        return f"{brand}{word}.com"
    elif kind < 0.7:
        chars = list(brand)
        pos = random.randint(0, len(chars))
        chars.insert(pos, random.choice(string.ascii_lowercase))
        return f"{''.join(chars)}.com"
    else:
        return f"{brand}-secure-{random.choice(['login','verify'])}.{random.choice(['tk','xyz','net'])}"


def run_simulated(model_path, feed_path, interval_seconds=3, stop_event=None, threshold=0.5):
    clf, feature_names = _load_model(model_path)
    Path(feed_path).touch(exist_ok=True)
    print(f"[simulated] writing flagged domains to {feed_path}")
    while stop_event is None or not stop_event.is_set():
        attack_like = random.random() < 0.35
        domain = _random_domain(attack_like)
        if _is_brand_relevant(domain):
            prob, feats = _score_domain(clf, feature_names, domain)
            if prob >= threshold:
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "domain": domain,
                    "phishing_probability": round(prob, 4),
                    "brand_edit_distance": feats["brand_edit_distance"],
                    "suspicious_tld": bool(feats["suspicious_tld"]),
                    "source": "simulated",
                }
                _append_to_feed(feed_path, record)
        time.sleep(interval_seconds)


# ---------------------------------------------------------------------------
# Live mode (real Certificate Transparency stream, needs internet)
# ---------------------------------------------------------------------------

def _write_status(feed_path, **fields):
    status_path = Path(feed_path).with_name("ct_status.json")
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), **fields}
    with open(status_path, "w") as f:
        json.dump(payload, f)


def run_live(model_path, feed_path, threshold=0.5, stop_event=None):
    if certstream is None:
        raise RuntimeError("certstream is not installed. pip install certstream")
    clf, feature_names = _load_model(model_path)
    Path(feed_path).touch(exist_ok=True)
    _write_status(feed_path, state="connecting")

    def on_open():
        # Socket handshake succeeded. Doesn't mean data has arrived yet --
        # that's confirmed separately on the first certificate_update.
        _write_status(feed_path, state="socket_open")

    def on_error(ex):
        # certstream's own run_forever() swallows connection failures and
        # retries internally -- it does NOT raise, so this callback is the
        # only place we actually learn a connection attempt failed.
        _write_status(feed_path, state="error", error=str(ex))

    def callback(message, context):
        if stop_event is not None and stop_event.is_set():
            raise KeyboardInterrupt("stop requested")
        if message.get("message_type") != "certificate_update":
            return
        # Receiving any certificate_update proves data is actually flowing,
        # even before a brand-relevant / flagged domain shows up.
        _write_status(feed_path, state="connected")
        try:
            domains = message["data"]["leaf_cert"].get("all_domains", [])
        except (KeyError, TypeError):
            return
        for domain in domains:
            domain = domain.lstrip("*.")
            if not domain or not _is_brand_relevant(domain):
                continue
            prob, feats = _score_domain(clf, feature_names, domain)
            if prob >= threshold:
                record = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "domain": domain,
                    "phishing_probability": round(prob, 4),
                    "brand_edit_distance": feats["brand_edit_distance"],
                    "suspicious_tld": bool(feats["suspicious_tld"]),
                    "source": "live_ct_log",
                }
                _append_to_feed(feed_path, record)

    print("[live] connecting to Certificate Transparency stream via certstream...")
    try:
        certstream.listen_for_events(
            callback, url="wss://certstream.calidog.io/", on_open=on_open, on_error=on_error
        )
    except KeyboardInterrupt:
        print("[live] certstream listener stopped")


if __name__ == "__main__":
    # Defaults are computed relative to this file's own location (not the
    # current working directory), so `python src/ct_feed_worker.py` works
    # whether it's invoked from the repo root (as the README does) or from
    # inside src/.
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(_REPO_ROOT / "models" / "model.joblib"))
    parser.add_argument("--feed", default=str(_REPO_ROOT / "data" / "live_feed.jsonl"))
    parser.add_argument("--mode", choices=["simulated", "live"], default="simulated")
    parser.add_argument("--interval", type=float, default=3.0)
    args = parser.parse_args()

    if args.mode == "simulated":
        run_simulated(args.model, args.feed, interval_seconds=args.interval)
    else:
        run_live(args.model, args.feed)