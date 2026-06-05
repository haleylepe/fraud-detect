#!/usr/bin/env python3
"""
data/seed_db.py — Populate sentinel.db with synthetic cases for 8 fictional users.

Runs the real Detoxify + rules pipeline on every text so scores are genuine.
Claude explanations are skipped (stubbed) to keep seeding fast and free.

Usage (run from the project root):
    python data/seed_db.py           # append synthetic cases
    python data/seed_db.py --fresh   # delete existing seeded cases first, then re-seed
"""

import sys
import random
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from detoxify import Detoxify
from rules import run_rules
from database import init_db, save_case, get_conn, get_next_case_id

# ---------------------------------------------------------------------------
# Fictional users and their content submissions
# Each case: (text, analyst_action, source_label)
# ---------------------------------------------------------------------------

SUBMISSIONS: list[dict] = [
    # ── alice_j ── clean, legitimate product reviewer ─────────────────────
    {
        "user": "alice_j",
        "cases": [
            (
                "The product arrived exactly as described. Packaging was great and "
                "delivery was faster than expected. Would definitely buy from this seller again.",
                "approved",
                "product-review",
            ),
            (
                "Five stars! The seller was very responsive and the item quality is "
                "excellent. Highly recommend to anyone looking for this.",
                "approved",
                "product-review",
            ),
            (
                "Solid product — nothing special but does exactly what it says. "
                "Good value for the price point.",
                "approved",
                "product-review",
            ),
            (
                "A bit disappointed with the color. It is darker than the photos show. "
                "Otherwise fine and delivery was prompt.",
                "approved",
                "product-review",
            ),
            (
                "Excellent customer service. They resolved my sizing issue within 24 hours "
                "and sent a replacement at no charge. Very happy with the outcome.",
                "approved",
                "product-review",
            ),
        ],
    },

    # ── bob_k ── borderline — fraud keywords mixed with legit reviews ──────
    {
        "user": "bob_k",
        "cases": [
            (
                "CLICK HERE for exclusive deals on refurbished electronics. "
                "Act now before stock runs out! Limited time offer ends tonight.",
                "removed",
                "checkout-flag",
            ),
            (
                "This item is fake. I want a refund immediately or I will dispute "
                "the charge with my bank and file a formal complaint.",
                "escalated",
                "support-ticket",
            ),
            (
                "Great seller! Fast shipping and the item arrived in perfect condition. "
                "Very satisfied with the whole experience.",
                "approved",
                "product-review",
            ),
            (
                "Terrible quality. Not as described at all. Fake product, avoid this listing.",
                "pending",
                "product-review",
            ),
            (
                "Good value for money. Delivery was fast and packaging was solid. No complaints.",
                "approved",
                "product-review",
            ),
            (
                "Verify your account to claim your exclusive buyer protection guarantee. "
                "Click here now to secure your order.",
                "removed",
                "checkout-flag",
            ),
        ],
    },

    # ── carol_m ── high risk — persistent scammer / fraudulent listings ────
    {
        "user": "carol_m",
        "cases": [
            (
                "Send wire transfer to secure your order. Limited time offer. "
                "Act now before this deal expires!!",
                "removed",
                "checkout-flag",
            ),
            (
                "FREE MONEY guaranteed for all verified users. "
                "Click here to claim your reward NOW!",
                "removed",
                "checkout-flag",
            ),
            (
                "URGENT: Your account has been suspended. Verify your account "
                "immediately to restore full access to your funds.",
                "removed",
                "user-report",
            ),
            (
                "I hate this platform and everyone who works here. "
                "Absolute garbage service, complete waste of time.",
                "removed",
                "support-ticket",
            ),
            (
                "This product is completely fake and a total scam. AVOID at all costs!!!",
                "escalated",
                "product-review",
            ),
            (
                "Good price, arrived quickly. No issues with the order whatsoever.",
                "approved",
                "product-review",
            ),
            (
                "The seller refunded me promptly. Very professional and courteous handling.",
                "approved",
                "product-review",
            ),
        ],
    },

    # ── dave_r ── medium risk — excessive caps, angry but not scamming ─────
    {
        "user": "dave_r",
        "cases": [
            (
                "THIS PRODUCT IS ABSOLUTELY TERRIBLE AND I AM EXTREMELY ANGRY "
                "ABOUT THE SERVICE I RECEIVED. COMPLETE WASTE OF MONEY. NEVER AGAIN.",
                "escalated",
                "support-ticket",
            ),
            (
                "I WANT MY MONEY BACK RIGHT NOW. THIS IS UNACCEPTABLE AND "
                "COMPLETELY OUTRAGEOUS BEHAVIOR FROM YOUR COMPANY.",
                "escalated",
                "support-ticket",
            ),
            (
                "Fine product, works as expected. Nothing special but it gets the job done.",
                "approved",
                "product-review",
            ),
            (
                "Quick delivery, good packaging. Happy with my purchase overall.",
                "approved",
                "product-review",
            ),
            (
                "SCAM ALERT: This seller is dishonest. "
                "The product is fake and nothing like the listing photos showed.",
                "removed",
                "user-report",
            ),
        ],
    },

    # ── eve_s ── low risk — mostly clean, one frustrated message ──────────
    {
        "user": "eve_s",
        "cases": [
            (
                "Lovely item, exactly as described. Will definitely shop here again. "
                "Fast, reliable, and well packaged.",
                "approved",
                "product-review",
            ),
            (
                "Great experience overall. The product quality exceeded my expectations "
                "for this price point. Very impressed.",
                "approved",
                "product-review",
            ),
            (
                "You sent me the completely wrong color and now I have to return it. "
                "This is very frustrating and a waste of my time.",
                "pending",
                "support-ticket",
            ),
            (
                "Fast shipping, product works perfectly right out of the box. "
                "Highly recommend this seller.",
                "approved",
                "product-review",
            ),
        ],
    },

    # ── frank_t ── very high risk — aggressive, threatening, consistently toxic
    {
        "user": "frank_t",
        "cases": [
            (
                "I will sue this entire company. You are all incompetent thieves "
                "and I genuinely hate every one of you.",
                "escalated",
                "support-ticket",
            ),
            (
                "This is the worst piece of garbage I have ever wasted money on. "
                "A complete scam. Avoid this seller at all costs.",
                "removed",
                "product-review",
            ),
            (
                "Your customer service is absolutely worthless. I have never dealt "
                "with such incompetent and rude people in my entire life.",
                "escalated",
                "support-ticket",
            ),
            (
                "Refund my money NOW or I will destroy your reputation online "
                "and contact every media outlet I can find. You have been warned!!!",
                "escalated",
                "support-ticket",
            ),
            (
                "What a pathetic excuse for a product. The listing photos are fake "
                "and everyone involved in this is a liar.",
                "removed",
                "user-report",
            ),
            (
                "WIRE TRANSFER scam! This seller tried to get my banking information. "
                "FRAUD ALERT! Report this listing immediately!",
                "removed",
                "user-report",
            ),
        ],
    },

    # ── grace_w ── low risk — mostly clean, one accidental spam submission ─
    {
        "user": "grace_w",
        "cases": [
            (
                "Beautiful product, arrived well packaged and ahead of schedule. "
                "Exactly what I ordered, very pleased.",
                "approved",
                "product-review",
            ),
            (
                "Click here for a free bonus gift with every order! "
                "Guaranteed satisfaction or your money back!",
                "removed",
                "checkout-flag",
            ),
            (
                "Super fast delivery. The item is exactly as described and the "
                "build quality is genuinely impressive.",
                "approved",
                "product-review",
            ),
            (
                "Really happy with this purchase. The seller was friendly and "
                "communicated well at every step.",
                "approved",
                "product-review",
            ),
        ],
    },

    # ── henry_p ── medium-high risk — repeated fraud keyword submissions ───
    {
        "user": "henry_p",
        "cases": [
            (
                "Guaranteed free refund if not satisfied. Click here to verify "
                "your account and claim your exclusive reward now.",
                "removed",
                "checkout-flag",
            ),
            (
                "Wire transfer payment only accepted. Act now for a limited time discount. "
                "Urgent offer expires very soon, do not miss out.",
                "removed",
                "checkout-flag",
            ),
            (
                "Product is good quality. Delivered in three days with no damage. "
                "No issues at all, would buy again.",
                "approved",
                "product-review",
            ),
            (
                "CLICK HERE to get your exclusive discount code. "
                "This offer expires in 24 hours only, act now!",
                "removed",
                "checkout-flag",
            ),
            (
                "Very satisfied with the purchase. Seller was professional "
                "and responsive to all my questions.",
                "approved",
                "product-review",
            ),
            (
                "URGENT NOTICE: Your account requires immediate verification. "
                "Send wire transfer to secure your funds now before they are frozen.",
                "removed",
                "user-report",
            ),
        ],
    },
]


# ---------------------------------------------------------------------------
# Severity helper (mirrors main.py logic exactly)
# ---------------------------------------------------------------------------

def _get_severity(verdict: str, toxicity: float, rule_count: int) -> str:
    if verdict == "CLEAN":
        return "clean"
    if toxicity > 0.7 or (toxicity > 0.4 and rule_count >= 2) or rule_count >= 3:
        return "high"
    if toxicity > 0.4 or rule_count >= 1:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    random.seed(42)  # reproducible timestamps
    fresh = "--fresh" in sys.argv

    init_db()

    if fresh:
        with get_conn() as conn:
            deleted = conn.execute(
                "DELETE FROM cases WHERE submitted_by IS NOT NULL"
            ).rowcount
        print(f"[seed] Cleared {deleted} existing seeded case(s).\n")

    # Flatten all texts so we can batch-predict with Detoxify in one call
    flat: list[tuple[str, str, str, str]] = []  # (user, text, action, source)
    for block in SUBMISSIONS:
        for text, action, source in block["cases"]:
            flat.append((block["user"], text, action, source))

    print(f"Loading Detoxify model…")
    model = Detoxify("original")
    print(f"Detoxify ready. Running batch prediction on {len(flat)} texts…\n")

    all_texts = [t for _, t, _, _ in flat]
    batch = model.predict(all_texts)  # dict of {score_name: [float, ...]}

    now = datetime.now(timezone.utc)

    # Generate timestamps: spread across the past 7 days.
    # We assign per-user windows so each user's submissions cluster naturally.
    # Sort them within each user group so earlier list = older date.
    user_indices: dict[str, list[int]] = {}
    for i, (user, _, _, _) in enumerate(flat):
        user_indices.setdefault(user, []).append(i)

    timestamps: list[datetime] = [now] * len(flat)
    for user, idxs in user_indices.items():
        n = len(idxs)
        # Generate N sorted random offsets across 7 days (in seconds, oldest first)
        offsets = sorted(random.uniform(0.5 * 86400, 6.9 * 86400) for _ in range(n))
        for idx, secs_ago in zip(idxs, offsets):
            jitter = random.uniform(-3600, 3600)
            timestamps[idx] = now - timedelta(seconds=secs_ago + jitter)

    saved = 0
    for i, (user, text, action, source) in enumerate(flat):
        scores = {k: round(float(v[i]), 4) for k, v in batch.items()}
        toxicity = scores["toxicity"]

        triggered_rules = run_rules(text, detoxify_toxicity=toxicity)
        verdict = "FLAGGED" if (toxicity > 0.5 or len(triggered_rules) > 0) else "CLEAN"
        severity = _get_severity(verdict, toxicity, len(triggered_rules))
        case_id = get_next_case_id()

        case = {
            "id": case_id,
            "verdict": verdict,
            "severity": severity,
            "text_preview": text[:120] + ("…" if len(text) > 120 else ""),
            "detoxify_scores": scores,
            "triggered_rules": triggered_rules,
            "explanation": "[Seeded — no LLM explanation generated]",
            "action": action,
            "created_at": timestamps[i].isoformat(),
            "source_label": source,
            "submitted_by": user,
        }
        save_case(case)
        saved += 1

        icon = "🔴" if verdict == "FLAGGED" else "🟢"
        rules_str = ",".join(r["rule"] for r in triggered_rules) or "—"
        print(
            f"  {icon} {case_id}  {user:<10}  tox={toxicity:.2f}  "
            f"{verdict:<7} {severity:<7}  [{action:<9}]  rules=[{rules_str}]"
        )

    print(f"\n✓ Seeded {saved} cases across {len(SUBMISSIONS)} users.")
    print(f"  DB: {ROOT / 'sentinel.db'}")


if __name__ == "__main__":
    main()
