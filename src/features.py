"""
Feature extraction for zero-day phishing detection.

DESIGN CONSTRAINT (this is the whole point of the project): every feature
here is derivable from the URL STRING ALONE, in milliseconds, with no
network call. No WHOIS age lookup, no reputation API, no blocklist check.

Why: those signals don't exist yet for a domain registered an hour ago.
A model that leans on them looks great on stale benchmarks and blind on
the URLs that actually matter -- the ones no one has reported yet.
"""

import re
import math
from collections import Counter
from urllib.parse import urlparse

import tldextract

# Use only the bundled offline snapshot of the public suffix list -- avoids a
# network fetch on every cold start (and matches the "no network dependency"
# design goal of this whole feature set).
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())

POPULAR_BRANDS = [
    "paypal", "google", "amazon", "microsoft", "apple", "netflix", "facebook",
    "instagram", "chase", "bankofamerica", "wellsfargo", "coinbase", "binance",
    "metamask", "dropbox", "linkedin", "twitter", "adobe", "outlook", "steam",
]

SUSPICIOUS_KEYWORDS = ["secure", "verify", "update", "login", "account", "confirm",
                        "signin", "billing", "suspended", "alert", "unlock", "reset",
                        "wallet", "support"]

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "xyz", "top", "buzz", "click", "loan", "work"}

IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


def shannon_entropy(s):
    if not s:
        return 0.0
    counts = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = curr
    return prev[-1]


def closest_brand_distance(domain):
    """Min edit distance to any popular brand name -- catches typosquats
    and homographs without needing a hardcoded blocklist of bad domains."""
    if not domain:
        return 99, None
    best = (99, None)
    for brand in POPULAR_BRANDS:
        d = levenshtein(domain.lower(), brand)
        if d < best[0]:
            best = (d, brand)
    return best


def extract_features(url: str) -> dict:
    url = url.strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    ext = _TLD_EXTRACTOR(url)
    domain = ext.domain
    subdomain = ext.subdomain
    suffix = ext.suffix
    full_host = parsed.netloc
    path = parsed.path or ""

    brand_dist, nearest_brand = closest_brand_distance(domain)
    # brand appearing in subdomain but NOT as the actual registered domain
    # e.g. paypal.secure-login.tk -> brand in subdomain, real domain is "secure-login"
    brand_in_subdomain_not_domain = any(
        b in subdomain.lower() and b != domain.lower() for b in POPULAR_BRANDS
    )

    feats = {
        "url_length": len(url),
        "host_length": len(full_host),
        "path_length": len(path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_subdomains": len([s for s in subdomain.split(".") if s]) if subdomain else 0,
        "has_ip_host": int(bool(IP_RE.match(full_host.split(":")[0]))),
        "is_https": int(parsed.scheme == "https"),
        "has_at_symbol": int("@" in url),
        "has_double_slash_in_path": int("//" in path),
        "suspicious_tld": int(suffix.split(".")[-1] in SUSPICIOUS_TLDS) if suffix else 0,
        "keyword_count": sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url.lower()),
        "domain_entropy": shannon_entropy(domain),
        "path_entropy": shannon_entropy(path),
        "digit_letter_ratio": (sum(c.isdigit() for c in domain) / max(len(domain), 1)),
        "brand_edit_distance": brand_dist,  # 0 = exact brand match, low = typosquat-ish
        "brand_in_subdomain_not_domain": int(brand_in_subdomain_not_domain),
        "domain_length": len(domain),
        "longest_word_length": max((len(w) for w in re.split(r"[-._/]", domain)), default=0),
        "port_specified": int(":" in full_host and not full_host.endswith(":80")),
    }
    return feats


FEATURE_NAMES = list(extract_features("https://example.com/").keys())


if __name__ == "__main__":
    for u in ["https://www.paypal.com/login",
              "http://paypa1-secure.tk/verify-account",
              "http://192.168.1.5/paypal/login.php",
              "https://myblog42.io/"]:
        print(u)
        print(extract_features(u))
        print()
