"""
database.py — SQLite persistence layer for Sentinel.

Uses Python's built-in sqlite3 — no extra dependencies needed.

Tables:
    cases    — one row per analyzed submission
    policies — single-row active community-guidelines policy
"""

import sqlite3
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent / "sentinel.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id              TEXT PRIMARY KEY,
    verdict         TEXT NOT NULL,
    severity        TEXT NOT NULL,
    text_preview    TEXT NOT NULL,
    detoxify_scores TEXT NOT NULL,
    triggered_rules TEXT NOT NULL,
    explanation     TEXT NOT NULL,
    action          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL,
    archived        INTEGER NOT NULL DEFAULT 0,
    feedback_tag    TEXT,
    feedback_note   TEXT,
    source_label    TEXT,
    submitted_by    TEXT
);

CREATE TABLE IF NOT EXISTS policies (
    id         INTEGER PRIMARY KEY,
    name       TEXT NOT NULL DEFAULT '',
    content    TEXT NOT NULL DEFAULT '',
    updated_at TEXT NOT NULL
);
"""

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        # Migrate existing databases
        for stmt in [
            "ALTER TABLE cases ADD COLUMN archived      INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE cases ADD COLUMN feedback_tag  TEXT",
            "ALTER TABLE cases ADD COLUMN feedback_note TEXT",
            "ALTER TABLE cases ADD COLUMN source_label  TEXT",
            "ALTER TABLE cases ADD COLUMN submitted_by  TEXT",
        ]:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError:
                pass  # column already exists
    print(f"[db] Database ready at {DB_PATH}")

# ---------------------------------------------------------------------------
# Case operations
# ---------------------------------------------------------------------------

def save_case(case: dict):
    """Insert a new case. case dict must match the cases table schema."""
    with get_conn() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO cases
            (id, verdict, severity, text_preview, detoxify_scores, triggered_rules,
             explanation, action, created_at, source_label, submitted_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            case["id"],
            case["verdict"],
            case["severity"],
            case["text_preview"],
            json.dumps(case["detoxify_scores"]),
            json.dumps(case["triggered_rules"]),
            case["explanation"],
            case.get("action", "pending"),
            case.get("created_at", datetime.now(timezone.utc).isoformat()),
            case.get("source_label"),
            case.get("submitted_by"),
        ))

def update_action(case_id: str, action: str):
    """Update the analyst action for a case."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE cases SET action = ? WHERE id = ?",
            (action, case_id)
        )

def load_all_cases() -> list[dict]:
    """Return all non-archived cases ordered by newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cases WHERE archived = 0 ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for row in rows:
        c = dict(row)
        c["detoxify_scores"] = json.loads(c["detoxify_scores"])
        c["triggered_rules"] = json.loads(c["triggered_rules"])
        result.append(c)
    return result

def archive_case(case_id: str):
    """Move a case to the archive (hidden from main queue, kept in DB)."""
    with get_conn() as conn:
        conn.execute("UPDATE cases SET archived = 1 WHERE id = ?", (case_id,))

def unarchive_case(case_id: str):
    """Restore an archived case to the main queue."""
    with get_conn() as conn:
        conn.execute("UPDATE cases SET archived = 0 WHERE id = ?", (case_id,))

def load_archived_cases() -> list[dict]:
    """Return all archived cases ordered by newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cases WHERE archived = 1 ORDER BY created_at DESC"
        ).fetchall()
    result = []
    for row in rows:
        c = dict(row)
        c["detoxify_scores"] = json.loads(c["detoxify_scores"])
        c["triggered_rules"] = json.loads(c["triggered_rules"])
        result.append(c)
    return result

def set_case_feedback(case_id: str, tag: str | None, note: str):
    """Store analyst feedback (tag + note) on a case."""
    with get_conn() as conn:
        conn.execute(
            "UPDATE cases SET feedback_tag = ?, feedback_note = ? WHERE id = ?",
            (tag or None, note or None, case_id),
        )

def load_feedback_cases(limit: int = 5) -> list[dict]:
    """Return the most recent cases with analyst feedback, for prompt calibration."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM cases WHERE feedback_tag IS NOT NULL ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    result = []
    for row in rows:
        c = dict(row)
        c["detoxify_scores"] = json.loads(c["detoxify_scores"])
        c["triggered_rules"] = json.loads(c["triggered_rules"])
        result.append(c)
    return result

def delete_case(case_id: str):
    """Permanently delete a case by ID."""
    with get_conn() as conn:
        conn.execute("DELETE FROM cases WHERE id = ?", (case_id,))

def get_analytics() -> dict:
    """Aggregate analytics data across all cases (active + archived)."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT verdict, severity, action, created_at, triggered_rules, submitted_by "
            "FROM cases"
        ).fetchall()

    cases = [dict(r) for r in rows]
    for c in cases:
        c["triggered_rules"] = json.loads(c["triggered_rules"])

    total = len(cases)

    # Volume timeline — last 30 days, flagged vs clean per day
    daily: dict = defaultdict(lambda: {"flagged": 0, "clean": 0})
    for c in cases:
        date = c["created_at"][:10]
        key = "flagged" if c["verdict"] == "FLAGGED" else "clean"
        daily[date][key] += 1
    sorted_dates = sorted(daily.keys())[-30:]
    timeline = [{"date": d, "flagged": daily[d]["flagged"], "clean": daily[d]["clean"]}
                for d in sorted_dates]

    # Rule frequency
    rule_counts: Counter = Counter()
    for c in cases:
        for r in c["triggered_rules"]:
            rule_counts[r["rule"]] += 1
    rule_frequency = [{"rule": k, "count": v} for k, v in rule_counts.most_common(8)]

    # Severity distribution
    sev = Counter(c["severity"] for c in cases)
    severity_distribution = {k: sev.get(k, 0) for k in ("high", "medium", "low", "clean")}

    # Analyst action breakdown
    act = Counter(c["action"] for c in cases)
    action_distribution = {k: act.get(k, 0) for k in ("pending", "approved", "removed", "escalated")}

    # Rule co-occurrence clusters (2+ rules together)
    combo_counts: Counter = Counter()
    for c in cases:
        rules = tuple(sorted(r["rule"] for r in c["triggered_rules"]))
        if len(rules) >= 2:
            combo_counts[rules] += 1
    rule_clusters = [
        {"rules": list(combo), "count": count}
        for combo, count in combo_counts.most_common(6)
    ]

    # User risk — aggregate per submitted_by
    user_map: dict = defaultdict(lambda: {
        "total": 0, "flagged": 0, "actions": [], "last_seen": "",
    })
    for c in cases:
        u = c.get("submitted_by")
        if not u:
            continue
        user_map[u]["total"] += 1
        if c["verdict"] == "FLAGGED":
            user_map[u]["flagged"] += 1
        if c["action"] != "pending":
            user_map[u]["actions"].append(c["action"])
        if c["created_at"] > user_map[u]["last_seen"]:
            user_map[u]["last_seen"] = c["created_at"]

    user_risk = []
    for name, u in sorted(
        user_map.items(),
        key=lambda kv: kv[1]["flagged"] / max(kv[1]["total"], 1),
        reverse=True,
    ):
        flag_rate = round(u["flagged"] / u["total"], 3) if u["total"] else 0
        user_risk.append({
            "user": name,
            "total": u["total"],
            "flagged": u["flagged"],
            "flag_rate": flag_rate,
            "last_action": u["actions"][-1] if u["actions"] else "pending",
            "last_seen": u["last_seen"][:10] if u["last_seen"] else "",
        })

    return {
        "total": total,
        "timeline": timeline,
        "rule_frequency": rule_frequency,
        "severity_distribution": severity_distribution,
        "action_distribution": action_distribution,
        "rule_clusters": rule_clusters,
        "user_risk": user_risk,
    }

def get_next_case_id() -> str:
    """Generate next sequential case ID like CASE-0042."""
    with get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
    return f"CASE-{str(count + 1).zfill(4)}"

# ---------------------------------------------------------------------------
# Policy operations
# ---------------------------------------------------------------------------

def save_policy(name: str, content: str):
    """Upsert the single active policy (always stored at id=1)."""
    updated_at = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO policies (id, name, content, updated_at)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name       = excluded.name,
                content    = excluded.content,
                updated_at = excluded.updated_at
        """, (name, content, updated_at))

def get_active_policy() -> dict | None:
    """Return the active policy dict, or None if no policy is set."""
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM policies WHERE id = 1").fetchone()
    if row is None:
        return None
    d = dict(row)
    return d if d["content"].strip() else None

def clear_policy():
    """Remove the active policy."""
    with get_conn() as conn:
        conn.execute("DELETE FROM policies WHERE id = 1")
