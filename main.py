"""
main.py — FastAPI application entry point for the Fraud Review Dashboard.

Exposes two endpoints:
  GET  /          → serves the frontend (static/index.html)
  POST /analyze   → runs Detoxify + rules + LLM explanation on submitted text

Run with:  uvicorn main:app --reload
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from detoxify import Detoxify

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Fraud Review Dashboard", version="0.1.0")

# Load Detoxify once at startup — CPU inference, ~1-2s first load
print("Loading Detoxify model (this takes a few seconds on first run)...")
_detoxify = Detoxify("original")
print("Detoxify ready.")

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str

class RuleResult(BaseModel):
    rule: str
    detail: str

class AnalyzeResponse(BaseModel):
    verdict: str                        # "FLAGGED" or "CLEAN"
    detoxify_scores: dict[str, float]
    triggered_rules: list[RuleResult]
    explanation: str
    text_preview: str                   # first 120 chars of input, for UI

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page dashboard."""
    if not INDEX_HTML.exists():
        return HTMLResponse(
            content="<h2>Frontend not built yet — run Session 4.</h2>",
            status_code=200,
        )
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Main analysis endpoint.

    Pipeline:
      1. Validate input
      2. Run Detoxify toxicity scores
      3. Run hand-written signal rules   (rules.py — wired in Session 2)
      4. Call Claude for explanation      (explainer.py — wired in Session 3)
      5. Return structured response
    """
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text field must not be empty")
    if len(text) > 10_000:
        raise HTTPException(status_code=400, detail="text exceeds 10,000 character limit")

    # --- Step 2: Detoxify ---
    raw_scores: dict = _detoxify.predict(text)
    # Detoxify returns numpy floats; convert to plain Python floats for JSON
    detoxify_scores = {k: round(float(v), 4) for k, v in raw_scores.items()}

    # --- Step 3: Signal rules (placeholder until Session 2) ---
    # Will be replaced with: from rules import run_rules; triggered = run_rules(text)
    triggered_rules: list[RuleResult] = []

    # --- Step 4: LLM explanation (placeholder until Session 3) ---
    # Will be replaced with: from explainer import generate_explanation
    explanation = (
        "Explanation generation will be available after Session 3. "
        "Check Detoxify scores and triggered rules above for preliminary signals."
    )

    # --- Step 5: Verdict ---
    toxicity_score = detoxify_scores.get("toxicity", 0.0)
    is_flagged = toxicity_score > 0.5 or len(triggered_rules) > 0
    verdict = "FLAGGED" if is_flagged else "CLEAN"

    return AnalyzeResponse(
        verdict=verdict,
        detoxify_scores=detoxify_scores,
        triggered_rules=triggered_rules,
        explanation=explanation,
        text_preview=text[:120] + ("…" if len(text) > 120 else ""),
    )


# ---------------------------------------------------------------------------
# Dev runner (optional — prefer `uvicorn main:app --reload`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
