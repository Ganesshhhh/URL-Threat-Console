# URL Threat Console

A phishing URL classifier built around one constraint: every feature is
derived from the URL string alone, with no blocklist lookup or domain-age
check. That gap — catching a domain before it's been reported anywhere —
is the actual target of this project, not just another 95%-accuracy
tutorial classifier.

## The four things this does

1. **Score a URL** — paste a URL, get a flagged/not-flagged verdict with a
   SHAP-based explanation of exactly which features drove the score.
2. **Model comparison** — Logistic Regression, Random Forest, and XGBoost
   were all trained and evaluated on the same data. XGBoost was chosen
   based on adversarial red-team catch rate (80.6%), not held-out accuracy
   (all three scored ~0.99–1.00 there, which is why that number alone
   wasn't good enough to choose between them). The comparison is shown in
   the app, not just claimed in a doc.
3. **Live monitoring** — a background worker scores newly-seen domains and
   writes anything flagged to a feed the UI polls automatically. Important
   distinction: **this automates scoring and alerting, not retraining.** A
   Certificate Transparency log tells you a domain exists; it never tells
   you whether that domain turned out to be phishing, so there's no label
   to retrain a supervised model against. The model's weights are fixed at
   training time — what runs continuously is inference.
4. **Phantom squatting** — a distinct, newer attack pattern: LLMs
   frequently hallucinate plausible domains when asked for a brand's URL,
   and attackers register the hallucinated domain before it's ever
   visited. See "Phantom squatting" below for sourcing.

## Project structure

```
phishing-zero-day-detector/
├── data/
│   ├── load_openphish_tranco.py        # loads OpenPhish (pyopdb) + Tranco data into train.csv
│   ├── generate_hard_examples.py       # OPTIONAL: augments with harder synthetic cases
│   ├── model_comparison_results.json   # LR vs RF vs XGBoost results, read by the app
│   └── live_feed.jsonl                 # written by ct_feed_worker.py, read by the app
├── src/
│   ├── features.py          # URL-only feature extraction
│   ├── train.py              # trains the production XGBoost model
│   ├── compare_models.py     # LR vs RF vs XGBoost, saves model.joblib + comparison JSON
│   ├── adversarial.py        # red-team evaluation against 6 attack techniques
│   ├── explain.py            # SHAP explanation CLI for a single URL
│   ├── predict.py            # quick CLI scorer
│   ├── ct_feed_worker.py     # live-monitoring worker (simulated + real CT log modes)
│   └── phantom_squat.py      # phantom-squat scoring + real LLM-query harness
├── app/
│   └── streamlit_app.py      # the dashboard — all four tabs above
└── models/
    └── model.joblib           # the single production model (XGBoost)
```

## Setup and run

```bash
pip install -r requirements.txt

# 1. Get the training data: OpenPhish (via pyopdb / live feed) for phishing URLs
#    and Tranco crawler for benign URLs.
python data/load_openphish_tranco.py --out data/train.csv

# 2. Compare models and pick the winner (writes models/model.joblib
#    and data/model_comparison_results.json)
cd src
python compare_models.py --data ../data/train.csv

# 3. Launch the app
cd ..
streamlit run app/streamlit_app.py
```

If you only want to retrain the chosen model without re-running the full
comparison: `python src/train.py`.

## Training data: OpenPhish (pyopdb) & Tranco Crawler

`data/load_openphish_tranco.py` builds the training dataset:

1. **Phishing URLs (label = 1)**: Sourced from OpenPhish via the `pyopdb` library / SQLite database and the live OpenPhish feed (`https://openphish.com/feed.txt`).
2. **Benign URLs (label = 0)**: Sourced from top-ranked domains collected via the `tranco` crawler package, converted into valid standard URLs.

`--max_rows` (default 20,000) caps the dataset for faster iteration; pass
`--max_rows 0` to use all available URLs.

`data/generate_hard_examples.py` still exists as an **optional** add-on —
if the red-team results on real data reveal a specific gap, it can append
harder synthetic cases on top of the real data. It is not a substitute for
real training data and is not run by default.

## Live monitoring: simulated vs real

The "Live monitoring" tab defaults to a simulated feed — a background
thread that generates synthetic newly-registered-looking domains and
scores them, so the auto-updating UI works with zero setup. Switching to
"Live Certificate Transparency stream" requires:

```bash
pip install certstream
python src/ct_feed_worker.py --mode live --model models/model.joblib --feed data/live_feed.jsonl
```

This needs outbound internet access to `certstream.calidog.io`. Every
publicly-trusted HTTPS certificate is logged there within minutes of
issuance, including ones issued to phishing domains before anyone's
reported them — that's the actual "zero-day" part of this project.

## Phantom squatting

Large language models frequently hallucinate domain names for real
organizations, generating URLs that look authoritative but don't exist.
When someone (or an autonomous agent) asks an LLM for a brand's URL, the
model may present a hallucinated address as fact; if an attacker has
already registered that address, the person lands on a phishing page with
no email, ad, or search engine involved. Security researchers have found
measurable lead time here — one monitoring pipeline reported 18 to 51
days between detecting a hallucination and the domain actually being
registered by an attacker, a window reactive blocklists don't offer.
(Cloud Security Alliance research note, July 2026:
https://labs.cloudsecurityalliance.org/research/csa-research-note-phantom-squatting-ai-hallucinated-domains/)

This is a different generative process than human typosquatting: an LLM
tends to produce a clean, grammatically-plausible brand+word combination
on an ordinary TLD, not a fat-finger typo. `src/phantom_squat.py` scores
domains against that pattern and includes a real LLM-query harness
(`query_llm_live()`) — point it at your own `ANTHROPIC_API_KEY` to ask a
model what URL it associates with a brand and log genuine hallucinations.

**Honesty note:** this sandbox has no LLM API key configured, so the demo
table in the app is hand-built to illustrate the scoring pattern, clearly
labeled as such. It is not captured from a real API call.

## Known limitations

- **URL shorteners defeat every model tested (0% catch rate across LR,
  RF, and XGBoost).** A shortened link carries no structural signal for
  any URL-only classifier. Fixing this means resolving the redirect
  first, which reintroduces the network dependency this project was
  built to avoid — a real tradeoff, not an oversight.
- **Synthetic training data is easier than the real world** — no longer
  applicable now that training runs on real PhiUSIIL data, but worth
  remembering if you ever fall back to `generate_hard_examples.py` alone:
  a 100% held-out accuracy number is a reason for suspicion, not a brag.
- **The brand list is hardcoded to 20 names.** A real deployment needs a
  much larger, maintained list.
- **The live-monitoring tab automates alerting, not learning.** See the
  distinction above — worth restating because it's the part most likely
  to be misread as "the model retrains itself."
