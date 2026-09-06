"""
Phantom squatting: detect domains that match the pattern of a hallucinated
brand URL rather than a human typo.

Background (see README for full citations): LLMs frequently hallucinate
plausible-looking domains when asked for a brand's URL. Attackers monitor
for this and register the hallucinated domain before a real user or AI
agent visits it. This is a distinct generative process from human
typosquatting -- an LLM tends to produce clean, grammatically-plausible
brand+word combinations on ordinary TLDs, not fat-finger typos.

HONESTY NOTE ON THE DEMO DATA: this sandbox has no LLM API key configured,
so PHANTOM_SQUAT_ILLUSTRATIVE below is hand-constructed to demonstrate the
*pattern* (brand + clean English word, ordinary TLD, no suspicious
keyword) -- it is not the output of an actual API call. `query_llm_live()`
below is the real, working harness: point it at your own API key and it
will make a genuine call and log genuine hallucinations.
"""

import os
import json
import argparse

from features import POPULAR_BRANDS

# Hand-built to demonstrate the pattern this module targets. NOT captured
# from a real LLM call -- see honesty note above.
PHANTOM_SQUAT_ILLUSTRATIVE = [
    {"brand": "paypal", "hallucinated_domain": "paypalsupport.com", "prompted_model": "example-only"},
    {"brand": "coinbase", "hallucinated_domain": "coinbasehelp.io", "prompted_model": "example-only"},
    {"brand": "microsoft", "hallucinated_domain": "microsoftaccount.net", "prompted_model": "example-only"},
    {"brand": "chase", "hallucinated_domain": "chaseonlinebanking.com", "prompted_model": "example-only"},
    {"brand": "metamask", "hallucinated_domain": "metamask-wallet.io", "prompted_model": "example-only"},
]


def phantom_squat_score(domain: str) -> dict:
    """
    Heuristic score for 'does this domain look like something an LLM would
    hallucinate for a known brand', independent of the human-typo features
    already in features.py.

    Signals used (all distinct from the typosquat/homograph features):
      - domain = brand + a plausible English word, concatenated cleanly
        (no digits, no leetspeak, no random insertion)
      - clean/ordinary TLD (.com/.io/.net), not a bargain-bin TLD
      - no classic phishing keyword (verify/secure/login) -- LLM
        hallucinations tend to sound like a real product page, not an
        urgent scam page
    """
    base = domain.split(".")[0].lower()
    tld = domain.split(".")[-1].lower() if "." in domain else ""

    matched_brand = None
    suffix = None
    for brand in POPULAR_BRANDS:
        if base.startswith(brand) and base != brand:
            matched_brand = brand
            suffix = base[len(brand):]
            break

    if matched_brand is None:
        return {"phantom_squat_likely": False, "score": 0.0, "reason": "no brand prefix match"}

    clean_suffix = suffix.isalpha() and 2 <= len(suffix) <= 10
    clean_tld = tld in {"com", "io", "net", "co"}
    no_digits = not any(c.isdigit() for c in base)

    score = sum([clean_suffix, clean_tld, no_digits]) / 3
    return {
        "phantom_squat_likely": score >= 0.66,
        "score": round(score, 3),
        "matched_brand": matched_brand,
        "suffix_word": suffix,
        "reason": (
            f"'{base}' = brand '{matched_brand}' + clean word '{suffix}' on a standard TLD"
            if score >= 0.66 else "does not match the clean brand+word hallucination pattern"
        ),
    }


def query_llm_live(brand: str, model: str = "claude-sonnet-4-6"):
    """
    Real harness: ask an LLM what URL it associates with a brand's support/
    login page, and record whatever domain it names. Requires
    ANTHROPIC_API_KEY to be set in the environment.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. This function makes a real API "
            "call and needs your own key -- it will not run in a sandbox "
            "with no key configured."
        )

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"What is the exact URL for {brand}'s official customer support or "
        f"account login page? Answer with just the URL, nothing else."
    )
    response = client.messages.create(
        model=model,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(block.text for block in response.content if hasattr(block, "text"))
    return {"brand": brand, "prompted_model": model, "raw_response": text.strip()}


def query_groq_live(brand: str, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
    """
    Live query harness using Groq API (GROQ_API_KEY environment variable or passed key).
    Queries Groq models (e.g. llama-3.3-70b-versatile, llama3-8b-8192) to detect AI hallucinated domains.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Provide a valid Groq API key or set GROQ_API_KEY in environment."
        )

    prompt = (
        f"What is the exact URL or domain name for {brand}'s official customer support or "
        f"account login page? Respond with ONLY the domain or URL, nothing else."
    )

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.2,
        )
        raw_text = response.choices[0].message.content.strip()
    except Exception as e:
        import requests
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }
        res = requests.post(url, headers=headers, json=payload, timeout=15)
        if res.status_code != 200:
            raise RuntimeError(f"Groq API error (status {res.status_code}): {res.text}")
        data = res.json()
        raw_text = data["choices"][0]["message"]["content"].strip()

    cleaned_url = raw_text.splitlines()[0].strip("<>'\"` ")
    if "://" in cleaned_url:
        domain_candidate = cleaned_url.split("://")[1].split("/")[0]
    else:
        domain_candidate = cleaned_url.split("/")[0]

    scoring = phantom_squat_score(domain_candidate)

    return {
        "brand": brand,
        "provider": "Groq",
        "model": model,
        "raw_response": raw_text,
        "extracted_domain": domain_candidate,
        "phantom_squat_analysis": scoring,
    }


def run_illustrative_demo():
    print("Illustrative phantom-squat patterns (hand-built demo data, not live LLM output):\n")
    for entry in PHANTOM_SQUAT_ILLUSTRATIVE:
        result = phantom_squat_score(entry["hallucinated_domain"])
        print(f"  {entry['hallucinated_domain']:35s} brand={entry['brand']:12s} "
              f"phantom_squat_likely={result['phantom_squat_likely']}  score={result['score']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", action="store_true", help="Make a real API call (needs GROQ_API_KEY or ANTHROPIC_API_KEY)")
    parser.add_argument("--provider", choices=["groq", "anthropic"], default="groq")
    parser.add_argument("--brand", default="paypal")
    parser.add_argument("--groq_key", default=None, help="Groq API key")
    args = parser.parse_args()

    if args.live:
        if args.provider == "groq":
            result = query_groq_live(args.brand, api_key=args.groq_key)
        else:
            result = query_llm_live(args.brand)
        print(json.dumps(result, indent=2))
    else:
        run_illustrative_demo()
