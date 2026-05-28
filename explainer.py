"""
explainer.py — LLM-powered explanation generator for the Fraud Review Dashboard.

Calls the Anthropic Claude API to produce a plain-English, 2-3 sentence
explanation of why a piece of content was flagged. Written for a non-technical
fraud analyst audience — no jargon, no score dumps, just clear reasoning.

Public API:
    generate_explanation(
        text: str,
        detoxify_scores: dict[str, float],
        triggered_rules: list[dict],
    ) -> str

The Anthropic API key is read from the environment variable ANTHROPIC_API_KEY.
Set it in a .env file or export it in your shell before starting the server.
"""

import os
import anthropic
from dotenv import load_dotenv

load_dotenv()  # picks up .env file if present

# ---------------------------------------------------------------------------
# Client — instantiated once at module load
# ---------------------------------------------------------------------------

_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

_MODEL = "claude-sonnet-4-20250514"

# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_prompt(
    text: str,
    detoxify_scores: dict[str, float],
    triggered_rules: list[dict],
) -> str:
    """
    Build the user message sent to Claude.
    Includes the original text, Detoxify scores, and triggered rules
    so the model has full context for its explanation.
    """
    # Format detoxify scores as a readable list
    score_lines = "\n".join(
        f"  - {k}: {v:.4f}" for k, v in detoxify_scores.items()
    )

    # Format triggered rules (or note that none fired)
    if triggered_rules:
        rule_lines = "\n".join(
            f"  - {r['rule']}: {r['detail']}" for r in triggered_rules
        )
    else:
        rule_lines = "  (none)"

    return f"""You are a fraud and abuse analyst assistant. A piece of content has been submitted to our review system and the automated pipeline has produced the following signals. Your job is to write a 2-3 sentence plain-English explanation for a non-technical fraud analyst explaining *why* this content looks suspicious or problematic.

Be specific — reference the actual signals. Do not just say "this content is toxic." Explain what pattern or combination of signals makes it concerning. If the content seems mostly clean despite being flagged, say so honestly.

---

SUBMITTED TEXT:
\"\"\"{text}\"\"\"

DETOXIFY SCORES (0 = clean, 1 = toxic):
{score_lines}

TRIGGERED SIGNAL RULES:
{rule_lines}

---

Write your 2-3 sentence explanation now. Address it directly to the analyst, starting with what the most concerning signal is."""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_explanation(
    text: str,
    detoxify_scores: dict[str, float],
    triggered_rules: list[dict],
) -> str:
    """
    Call the Claude API and return a plain-English explanation string.

    Args:
        text:             The original submitted text.
        detoxify_scores:  Dict of Detoxify score names → float values.
        triggered_rules:  List of rule dicts (from rules.run_rules).

    Returns:
        A 2-3 sentence explanation string, or an error message string
        if the API call fails (so the dashboard still renders).
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )

    prompt = _build_prompt(text, detoxify_scores, triggered_rules)

    try:
        message = _client.messages.create(
            model=_MODEL,
            max_tokens=1000,
            system=(
                "You are a concise, accurate fraud analyst assistant. "
                "You write clear, jargon-free explanations for human reviewers. "
                "Never reproduce the submitted text verbatim in your explanation. "
                "Always be specific about which signals are most concerning."
            ),
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return message.content[0].text.strip()

    except anthropic.AuthenticationError:
        return "API key is invalid or expired. Check your ANTHROPIC_API_KEY."
    except anthropic.RateLimitError:
        return "Claude API rate limit reached. Try again in a few seconds."
    except Exception as exc:
        return f"Explanation unavailable: {exc}"


# ---------------------------------------------------------------------------
# Smoke test — run directly with: python explainer.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_text = "THIS PRODUCT IS ABSOLUTELY FAKE AND A TOTAL SCAM"
    sample_scores = {
        "toxicity": 0.5638,
        "severe_toxicity": 0.0008,
        "obscene": 0.0291,
        "threat": 0.001,
        "insult": 0.022,
        "identity_attack": 0.0016,
    }
    sample_rules = [
        {"rule": "EXCESSIVE_CAPS", "detail": "100% of letters are uppercase (threshold: 40%)"},
        {"rule": "KEYWORD_FLAG", "detail": "Matched keyword(s): fake"},
        {"rule": "SHORT_SUSPICIOUS", "detail": "9 words (threshold: <15) with toxicity score 0.56 (threshold: >0.3)"},
    ]

    print("Calling Claude API...")
    explanation = generate_explanation(sample_text, sample_scores, sample_rules)
    print(f"\nExplanation:\n{explanation}")
