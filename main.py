"""
main.py — FastAPI application for Sentinel Fraud Review Dashboard.

Endpoints:
  GET  /                    → serves the frontend
  POST /analyze             → analyze text, save to DB, return result
  POST /analyze/batch       → batch analyze (no LLM, no DB save)
  GET  /analytics           → aggregated analytics across all cases
  GET  /cases               → load all cases from DB
  POST   /cases/{id}/action    → update analyst decision in DB
  POST   /cases/{id}/feedback  → save analyst feedback tag + note
  GET    /cases/archived      → list archived cases
  POST   /cases/{id}/archive  → move case to archive
  POST   /cases/{id}/unarchive → restore case from archive
  DELETE /cases/{id}          → permanently delete a case
  GET  /policy              → get the active community-guidelines policy
  POST /policy              → save/update the policy
  DELETE /policy            → clear the policy
"""

import os
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from detoxify import Detoxify
from rules import run_rules
from explainer import generate_explanation
from database import (
    init_db, save_case, load_all_cases, update_action, get_next_case_id,
    save_policy, get_active_policy, clear_policy,
    delete_case, archive_case, unarchive_case, load_archived_cases,
    set_case_feedback, load_feedback_cases, get_analytics,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Sentinel — Fraud Review Dashboard", version="0.3.0")

print("Loading Detoxify model...")
_detoxify = Detoxify("original")
print("Detoxify ready.")

init_db()

STATIC_DIR = Path(__file__).parent / "static"
INDEX_HTML = STATIC_DIR / "index.html"

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str
    source_label: str | None = None
    submitted_by: str | None = None

class BatchAnalyzeRequest(BaseModel):
    texts: list[str]

class ActionRequest(BaseModel):
    action: str   # "approved" | "removed" | "escalated" | "pending"

class RuleResult(BaseModel):
    rule: str
    detail: str

class AnalyzeResponse(BaseModel):
    id: str
    verdict: str
    severity: str
    detoxify_scores: dict[str, float]
    triggered_rules: list[RuleResult]
    explanation: str
    text_preview: str
    action: str
    created_at: str

class BatchItemResponse(BaseModel):
    verdict: str
    detoxify_scores: dict[str, float]
    triggered_rules: list[RuleResult]
    explanation: str
    text_preview: str

class BatchAnalyzeResponse(BaseModel):
    results: list[BatchItemResponse]
    total: int
    flagged_count: int

class FeedbackRequest(BaseModel):
    tag: str | None = None
    note: str = ""

class PolicyRequest(BaseModel):
    name: str
    content: str

class PolicyResponse(BaseModel):
    name: str
    content: str
    updated_at: str

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_severity(verdict: str, toxicity: float, rule_count: int) -> str:
    if verdict == "CLEAN":
        return "clean"
    if toxicity > 0.7 or (toxicity > 0.4 and rule_count >= 2) or rule_count >= 3:
        return "high"
    if toxicity > 0.4 or rule_count >= 1:
        return "medium"
    return "low"

def _run_pipeline(text: str, include_explanation: bool = True) -> dict:
    raw_scores = _detoxify.predict(text)
    detoxify_scores = {k: round(float(v), 4) for k, v in raw_scores.items()}
    toxicity = detoxify_scores.get("toxicity", 0.0)

    raw_rules = run_rules(text, detoxify_toxicity=toxicity)
    triggered_rules = [{"rule": r["rule"], "detail": r["detail"]} for r in raw_rules]

    if include_explanation:
        policy = get_active_policy()
        policy_text = policy["content"] if policy else ""
        examples = load_feedback_cases(limit=5)
        explanation = generate_explanation(
            text, detoxify_scores, raw_rules,
            policy=policy_text, feedback_examples=examples,
        )
    else:
        explanation = "Batch mode — explanations skipped."

    verdict = "FLAGGED" if (toxicity > 0.5 or len(raw_rules) > 0) else "CLEAN"
    severity = _get_severity(verdict, toxicity, len(raw_rules))

    return {
        "verdict": verdict,
        "severity": severity,
        "detoxify_scores": detoxify_scores,
        "triggered_rules": triggered_rules,
        "explanation": explanation,
        "text_preview": text[:120] + ("…" if len(text) > 120 else ""),
    }

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not INDEX_HTML.exists():
        return HTMLResponse(content="<h2>Frontend not found.</h2>")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > 10_000:
        raise HTTPException(status_code=400, detail="text exceeds 10,000 character limit")

    result = _run_pipeline(text, include_explanation=True)
    case_id = get_next_case_id()
    created_at = datetime.now(timezone.utc).isoformat()

    case = {
        "id": case_id,
        "action": "pending",
        "created_at": created_at,
        "source_label": req.source_label,
        "submitted_by": req.submitted_by,
        **result,
    }
    save_case(case)

    return AnalyzeResponse(**case)


@app.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(req: BatchAnalyzeRequest):
    texts = [t.strip() for t in req.texts if t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="no non-empty texts provided")
    if len(texts) > 20:
        raise HTTPException(status_code=400, detail="batch limit is 20 texts")

    results = [_run_pipeline(t, include_explanation=False) for t in texts]
    flagged = sum(1 for r in results if r["verdict"] == "FLAGGED")

    return BatchAnalyzeResponse(
        results=[BatchItemResponse(**r) for r in results],
        total=len(results),
        flagged_count=flagged,
    )


@app.get("/analytics")
async def analytics():
    """Return aggregated analytics data across all cases."""
    return get_analytics()


@app.get("/cases")
async def get_cases():
    """Load all cases from the database — called on frontend startup."""
    return load_all_cases()


@app.post("/cases/{case_id}/action")
async def set_action(case_id: str, req: ActionRequest):
    """Update analyst decision for a case."""
    valid = {"approved", "removed", "escalated", "pending"}
    if req.action not in valid:
        raise HTTPException(status_code=400, detail=f"action must be one of {valid}")
    update_action(case_id, req.action)
    return {"id": case_id, "action": req.action}


@app.post("/cases/{case_id}/feedback", status_code=204)
async def save_feedback(case_id: str, req: FeedbackRequest):
    """Store analyst feedback (tag + note) on a case."""
    set_case_feedback(case_id, req.tag, req.note.strip())


@app.get("/cases/archived")
async def get_archived_cases():
    """Return all archived cases."""
    return load_archived_cases()


@app.post("/cases/{case_id}/archive", status_code=204)
async def archive_case_endpoint(case_id: str):
    """Archive a case (removes from queue, keeps in DB)."""
    archive_case(case_id)


@app.post("/cases/{case_id}/unarchive", status_code=204)
async def unarchive_case_endpoint(case_id: str):
    """Restore an archived case to the main queue."""
    unarchive_case(case_id)


@app.delete("/cases/{case_id}", status_code=204)
async def remove_case(case_id: str):
    """Permanently delete a case from the database."""
    delete_case(case_id)


@app.get("/policy")
async def get_policy():
    """Return the active community-guidelines policy, or empty fields if none set."""
    policy = get_active_policy()
    if policy is None:
        return {"name": "", "content": "", "updated_at": ""}
    return policy


@app.post("/policy", response_model=PolicyResponse)
async def set_policy(req: PolicyRequest):
    """Save or update the active community-guidelines policy."""
    save_policy(req.name.strip(), req.content.strip())
    return get_active_policy()


@app.delete("/policy", status_code=204)
async def delete_policy():
    """Clear the active policy."""
    clear_policy()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)