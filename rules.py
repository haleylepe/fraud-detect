"""
rules.py — Hand-written signal rules for the Fraud Review Dashboard.

Each rule is a fast, deterministic check that runs independently of Detoxify.
Rules are cheap to compute and easy to explain to analysts.

Rules:
    Original (4):
        EXCESSIVE_CAPS         — >40% uppercase letters
        KEYWORD_FLAG           — badwords + fraud-specific terms (hughsie/badwords)
        SHORT_SUSPICIOUS       — <15 words with toxicity score >0.3
        REPEATED_CHARS         — any character repeated 4+ times in a row

    New (6, informed by WildChat violation categories):
        URL_DETECTED           — contains a hyperlink (phishing/spam signal)
        PERSONAL_INFO_REQUEST  — asks for SSN, password, credit card, etc.
        THREAT_LANGUAGE        — direct threat phrases ("I will kill", "I'll hurt", etc.)
        SUBSTANCE_REFERENCE    — references to drug use methods or illegal substances
        EXCESSIVE_PUNCTUATION  — 3+ consecutive ! or ? (spam/aggression signal)
        SELF_HARM_REFERENCE    — language patterns associated with self-harm

Public API:
    run_rules(text: str, detoxify_toxicity: float = 0.0) -> list[dict]

Keyword list source: badwords by Richard Hughes
    https://github.com/hughsie/badwords (English list)
"""

import re
import urllib.request
from functools import lru_cache

# ---------------------------------------------------------------------------
# Keyword list
# ---------------------------------------------------------------------------

_BADWORDS_URL = "https://raw.githubusercontent.com/hughsie/badwords/main/en"

_FRAUD_TERMS = [
    "refund", "lawsuit", "fake", "bot", "click here", "wire transfer",
    "free money", "guaranteed", "act now", "limited time", "verify your account",
    "suspended", "urgent", "send money", "gift card", "bitcoin payment",
    "your account has been", "confirm your identity", "unusual activity",
]

_FALLBACK_BADWORDS = [
    "damn", "hell", "crap", "piss", "ass", "bastard", "bitch", "shit",
    "fuck", "cunt", "dick", "cock", "pussy", "asshole", "motherfucker",
    "nigger", "faggot", "retard", "whore", "slut",
]

# Substance-related terms (WildChat: Controlled/Regulated Substances — 1,257 cases)
_SUBSTANCE_TERMS = [
    r"\bsnort\b", r"\bshoot\s+up\b", r"\binject\b",
    r"\bmeth\b", r"\bheroin\b", r"\bcrack\b", r"\bfentanyl\b",
    r"\bcocaine\b", r"\bxanax\b", r"\bprescription.{0,20}abuse\b",
    r"\bget\s+high\b", r"\bget\s+stoned\b", r"\bdrug\s+dealer\b",
    r"\bbuy\s+drugs\b", r"\billegal\s+pills\b",
]

# Threat language patterns (WildChat: Violence + Threat — 2,110 + 130 cases)
_THREAT_PATTERNS = [
    r"\bi('ll|'m going to|will|want to|plan to)\s+(kill|murder|hurt|harm|attack|destroy|shoot|stab|beat)\s+(you|him|her|them|someone|people)\b",
    r"\b(kill|murder|shoot|stab|attack)\s+(you|him|her|them|everyone)\b",
    r"\byou('re| are)\s+(dead|going to die|going to pay)\b",
    r"\bwatch\s+your\s+back\b",
    r"\bi\s+know\s+where\s+you\s+live\b",
    r"\bi('ll| will)\s+find\s+you\b",
]

# Self-harm patterns (WildChat: Suicide and Self Harm — 758 cases)
_SELF_HARM_PATTERNS = [
    r"\b(want to|going to|thinking about|planning to)\s+(kill|hurt|harm)\s+(myself|me)\b",
    r"\b(suicide|suicidal|end\s+my\s+life|take\s+my\s+life)\b",
    r"\b(cut\s+myself|self.?harm|self.?hurt)\b",
    r"\bdon't\s+want\s+to\s+(live|be\s+here|exist)\s+anymore\b",
    r"\bno\s+reason\s+to\s+(live|go\s+on)\b",
]

# Personal info request patterns (WildChat: PII/Privacy category)
_PII_PATTERNS = [
    r"\b(social\s+security|ssn)\s*(number|#)?\b",
    r"\b(credit|debit)\s+card\s+(number|details|info)\b",
    r"\bbank\s+(account|routing)\s+(number|details)\b",
    r"\bpassword\b",
    r"\bpin\s+(number|code)\b",
    r"\bdate\s+of\s+birth\b",
    r"\bmother'?s\s+maiden\s+name\b",
    r"\bsend\s+(me\s+)?your\s+(address|location|number|photo|pic)\b",
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
        fetched = [
            line.strip().lower()
            for line in raw.splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"[rules] Loaded {len(fetched)} words from hughsie/badwords")
    except Exception as exc:
        print(f"[rules] Could not fetch badwords list ({exc}); using fallback")
        fetched = _FALLBACK_BADWORDS
    return list(set(fetched + _FRAUD_TERMS))


# ---------------------------------------------------------------------------
# Original rules (4)
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
    """KEYWORD_FLAG: matches bad words + fraud-specific terms."""
    keywords = _load_keyword_list()
    text_lower = text.lower()
    matched = []
    for kw in keywords:
        if " " in kw:
            if kw in text_lower:
                matched.append(kw)
        else:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                matched.append(kw)
    if matched:
        shown = matched[:3]
        extra = len(matched) - 3
        detail = f"Matched keyword(s): {', '.join(shown)}"
        if extra > 0:
            detail += f" (+{extra} more)"
        return {"rule": "KEYWORD_FLAG", "detail": detail}
    return None


def _check_short_suspicious(text: str, detoxify_toxicity: float = 0.0) -> dict | None:
    """SHORT_SUSPICIOUS: fewer than 15 words with toxicity score > 0.3."""
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
    """REPEATED_CHARS: any single character repeated 4+ times in a row."""
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
# New rules (6) — informed by WildChat violation categories
# ---------------------------------------------------------------------------

def _check_url_detected(text: str) -> dict | None:
    """
    URL_DETECTED: text contains a hyperlink.
    Strong signal for phishing, spam, and malware distribution.
    """
    match = re.search(r"https?://[^\s]+", text)
    if match:
        url = match.group(0)
        display = url[:60] + ("…" if len(url) > 60 else "")
        return {
            "rule": "URL_DETECTED",
            "detail": f"Contains a hyperlink: {display}",
        }
    return None


def _check_personal_info_request(text: str) -> dict | None:
    """
    PERSONAL_INFO_REQUEST: asks for sensitive personal information.
    Covers SSN, passwords, bank details, and other PII.
    """
    text_lower = text.lower()
    for pattern in _PII_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            return {
                "rule": "PERSONAL_INFO_REQUEST",
                "detail": f"Requests sensitive personal information: '{match.group(0).strip()}'",
            }
    return None


def _check_threat_language(text: str) -> dict | None:
    """
    THREAT_LANGUAGE: direct threatening phrases toward a person.
    """
    text_lower = text.lower()
    for pattern in _THREAT_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            snippet = match.group(0)[:60]
            return {
                "rule": "THREAT_LANGUAGE",
                "detail": f"Contains threatening language: '{snippet}'",
            }
    return None


def _check_substance_reference(text: str) -> dict | None:
    """
    SUBSTANCE_REFERENCE: references to illicit drug use methods or substances.
    Catches queries that Detoxify scores near 0 despite being harmful.
    """
    text_lower = text.lower()
    for pattern in _SUBSTANCE_TERMS:
        match = re.search(pattern, text_lower)
        if match:
            snippet = match.group(0)[:50]
            return {
                "rule": "SUBSTANCE_REFERENCE",
                "detail": f"References controlled substance use: '{snippet}'",
            }
    return None


def _check_excessive_punctuation(text: str) -> dict | None:
    """
    EXCESSIVE_PUNCTUATION: 3+ consecutive ! or ? marks.
    Strong signal for spam, aggressive messaging, or manipulative urgency.
    """
    match = re.search(r"[!?]{3,}", text)
    if match:
        run = match.group(0)
        return {
            "rule": "EXCESSIVE_PUNCTUATION",
            "detail": f"Contains '{run}' — aggressive or spam-like punctuation pattern",
        }
    return None


def _check_self_harm_reference(text: str) -> dict | None:
    """
    SELF_HARM_REFERENCE: language associated with suicide or self-harm.
    Flag for human review — never auto-remove.
    """
    text_lower = text.lower()
    for pattern in _SELF_HARM_PATTERNS:
        match = re.search(pattern, text_lower)
        if match:
            return {
                "rule": "SELF_HARM_REFERENCE",
                "detail": "Contains language associated with self-harm or suicidal ideation — requires sensitive human review",
            }
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_rules(text: str, detoxify_toxicity: float = 0.0) -> list[dict]:
    """
    Run all 10 signal rules against the input text.

    Args:
        text:               The raw input string to analyze.
        detoxify_toxicity:  The toxicity score from Detoxify (0–1).
                            Pass this in so SHORT_SUSPICIOUS can use it.

    Returns:
        List of triggered rule dicts with 'rule' and 'detail' keys.
        Empty list means no rules fired.
    """
    _load_keyword_list()

    checkers = [
        # Original rules
        _check_excessive_caps(text),
        _check_keyword_flag(text),
        _check_short_suspicious(text, detoxify_toxicity),
        _check_repeated_chars(text),
        # New rules
        _check_url_detected(text),
        _check_personal_info_request(text),
        _check_threat_language(text),
        _check_substance_reference(text),
        _check_excessive_punctuation(text),
        _check_self_harm_reference(text),
    ]

    return [r for r in checkers if r is not None]


# ---------------------------------------------------------------------------
# Smoke test — run directly with: python rules.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("THIS PRODUCT IS ABSOLUTELY FAKE AND A TOTAL SCAM", 0.5),
        ("Click here for your free money wire transfer refund now", 0.1),
        ("I will kill you, you're dead, watch your back!!!", 0.8),
        ("How do I snort my prescription benzos to get high faster?", 0.2),
        ("Send me your social security number and bank account details", 0.1),
        ("I want to kill myself, I have no reason to live anymore", 0.3),
        ("Check out this amazing deal https://totally-legit-site.com/free", 0.1),
        ("Noooooooo this is terribleeeeee WTF!!!", 0.2),
        ("This seems like a completely normal review.", 0.02),
    ]
    for text, score in tests:
        results = run_rules(text, detoxify_toxicity=score)
        print(f"\nText: {text[:70]!r}")
        if results:
            for r in results:
                print(f"  ✗ {r['rule']}: {r['detail'][:80]}")
        else:
            print("  ✓ No rules triggered")
