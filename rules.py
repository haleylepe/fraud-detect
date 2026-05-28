"""
rules.py — Hand-written signal rules for the Fraud Review Dashboard.

Each rule is a fast, deterministic check that runs before (and independently of)
the Detoxify model. Rules are cheap to compute and easy to explain to analysts.

Public API:
    run_rules(text: str) -> list[dict]
        Returns a list of triggered rules, each a dict with keys:
            rule   (str) — rule name in SCREAMING_SNAKE_CASE
            detail (str) — human-readable explanation of why it triggered

Keyword list source: badwords by Richard Hughes
    https://github.com/hughsie/badwords (English list)
    Fetched at import time; falls back to a hardcoded list if the fetch fails.
"""

import re
import urllib.request
from functools import lru_cache

# ---------------------------------------------------------------------------
# Keyword list — fetched from hughsie/badwords, English list
# ---------------------------------------------------------------------------

_BADWORDS_URL = (
    "https://raw.githubusercontent.com/hughsie/badwords/main/en"
)

# Fraud-specific terms that are not in the generic bad-word list
_FRAUD_TERMS = [
    "refund",
    "lawsuit",
    "fake",
    "bot",
    "click here",
    "wire transfer",
    "free money",
    "guaranteed",
    "act now",
    "limited time",
    "verify your account",
    "suspended",
    "urgent",
]

# Hardcoded fallback — a small but representative set
_FALLBACK_BADWORDS = [
    "damn", "hell", "crap", "piss", "ass", "bastard", "bitch", "shit",
    "fuck", "cunt", "dick", "cock", "pussy", "asshole", "motherfucker",
    "nigger", "faggot", "retard", "whore", "slut",
]


@lru_cache(maxsize=1)
def _load_keyword_list() -> list[str]:
    """
    Fetch the English bad-word list from hughsie/badwords.
    Returns merged list of bad words + fraud terms (all lowercase).
    Result is cached after first call so the HTTP request only happens once.
    """
    try:
        with urllib.request.urlopen(_BADWORDS_URL, timeout=5) as resp:
            raw = resp.read().decode("utf-8")
        # Each line is one word; skip comments (#) and blank lines
        fetched = [
            line.strip().lower()
            for line in raw.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"[rules] Loaded {len(fetched)} words from hughsie/badwords")
    except Exception as exc:
        print(f"[rules] Could not fetch badwords list ({exc}); using fallback")
        fetched = _FALLBACK_BADWORDS

    # Merge with fraud-specific terms, deduplicate
    combined = list(set(fetched + _FRAUD_TERMS))
    return combined


# ---------------------------------------------------------------------------
# Individual rule implementations
# ---------------------------------------------------------------------------

def _check_excessive_caps(text: str) -> dict | None:
    """EXCESSIVE_CAPS: more than 40% of letter characters are uppercase."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    if upper_ratio > 0.40:
        return {
            "rule": "EXCESSIVE_CAPS",
            "detail": f"{upper_ratio:.0%} of letters are uppercase (threshold: 40%)",
        }
    return None


def _check_keyword_flag(text: str) -> dict | None:
    """
    KEYWORD_FLAG: text contains a word from the merged bad-word + fraud-term list.
    Uses whole-word matching to avoid false positives on substrings.
    """
    keywords = _load_keyword_list()
    text_lower = text.lower()

    matched = []
    for kw in keywords:
        # Multi-word fraud terms: substring match is fine (e.g. "click here")
        if " " in kw:
            if kw in text_lower:
                matched.append(kw)
        else:
            # Single words: require word boundary to avoid e.g. "classic" → "ass"
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                matched.append(kw)

    if matched:
        # Show up to 3 matched terms in the detail string
        shown = matched[:3]
        extra = len(matched) - 3
        detail = f"Matched keyword(s): {', '.join(shown)}"
        if extra > 0:
            detail += f" (+{extra} more)"
        return {"rule": "KEYWORD_FLAG", "detail": detail}
    return None


def _check_short_suspicious(text: str, detoxify_toxicity: float = 0.0) -> dict | None:
    """
    SHORT_SUSPICIOUS: fewer than 15 words AND detoxify toxicity score > 0.3.
    This rule requires the toxicity score passed in from the caller.
    If score is not available (e.g. during evaluate.py), it checks length only
    when score > 0.3 is explicitly provided.
    """
    word_count = len(text.split())
    if word_count < 15 and detoxify_toxicity > 0.3:
        return {
            "rule": "SHORT_SUSPICIOUS",
            "detail": (
                f"{word_count} words (threshold: <15) with "
                f"toxicity score {detoxify_toxicity:.2f} (threshold: >0.3)"
            ),
        }
    return None


def _check_repeated_chars(text: str) -> dict | None:
    """REPEATED_CHARS: any single character repeated 4 or more times in a row."""
    match = re.search(r"(.)\1{3,}", text)
    if match:
        char = match.group(1)
        run_length = len(match.group(0))
        display = repr(char) if char == " " else f"'{char}'"
        return {
            "rule": "REPEATED_CHARS",
            "detail": f"Character {display} repeated {run_length} times in a row",
        }
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_rules(text: str, detoxify_toxicity: float = 0.0) -> list[dict]:
    """
    Run all signal rules against the input text.

    Args:
        text:               The raw input string to analyze.
        detoxify_toxicity:  The toxicity score from Detoxify (0–1).
                            Pass this in so SHORT_SUSPICIOUS can use it.

    Returns:
        List of triggered rule dicts, each with 'rule' and 'detail' keys.
        Empty list means no rules fired.
    """
    # Warm up the keyword list on first call (cached after that)
    _load_keyword_list()

    checkers = [
        _check_excessive_caps(text),
        _check_keyword_flag(text),
        _check_short_suspicious(text, detoxify_toxicity),
        _check_repeated_chars(text),
    ]

    return [result for result in checkers if result is not None]


# ---------------------------------------------------------------------------
# Quick smoke test — run directly with: python rules.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("HELLO THIS IS VERY SUSPICIOUS AND VERY LOUD AND CAPS", 0.1),
        ("Click here to get your free money wire transfer refund now", 0.2),
        ("I hate you", 0.85),
        ("Noooooooo you can't do thissssss", 0.1),
        ("This seems like a completely normal and reasonable review.", 0.05),
    ]
    for text, score in tests:
        results = run_rules(text, detoxify_toxicity=score)
        print(f"\nText: {text!r}")
        print(f"Score passed in: {score}")
        if results:
            for r in results:
                print(f"  ✗ {r['rule']}: {r['detail']}")
        else:
            print("  ✓ No rules triggered")
