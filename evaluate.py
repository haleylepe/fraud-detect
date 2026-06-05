"""
evaluate.py — Evaluation script for the Fraud Review Dashboard.

Runs the detection pipeline (Detoxify + rules, NO LLM calls) against labeled
datasets and prints precision, recall, F1, confusion matrices, and a
cross-domain comparison table.

Usage:
    # First time only — prepare the datasets:
    python data/prepare_data.py

    # Standard evaluation (Jigsaw + Toxic Chat):
    python evaluate.py

    # Extended evaluation including WildChat:
    python evaluate.py --wildchat

Datasets:
    data/jigsaw_sample.csv    — columns: comment_text, toxic (0/1)
    data/toxicchat_sample.csv — columns: user_input, toxicity (0/1)
    data/wildchat_sample.csv  — columns: comment_text, toxic (0/1)  [optional]
"""

import os
import sys
import pandas as pd
from detoxify import Detoxify
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from rules import run_rules

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DETOXIFY_THRESHOLD = 0.5   # flag if detoxify toxicity score exceeds this
DATA_DIR = "data"

# ---------------------------------------------------------------------------
# Model — loaded once at import time
# ---------------------------------------------------------------------------

print("Loading Detoxify model...")
_model = Detoxify("original")
print("Ready.\n")


# ---------------------------------------------------------------------------
# Core pipeline (no LLM — too slow and costly for batch eval)
# ---------------------------------------------------------------------------

def predict(text: str) -> int:
    """
    Returns 1 (FLAGGED) or 0 (CLEAN) for a single text input.
    Mirrors the logic in main.py but skips the explanation step.
    """
    try:
        scores = _model.predict(str(text))
        toxicity = float(scores["toxicity"])
        triggered = run_rules(str(text), detoxify_toxicity=toxicity)
        return 1 if (toxicity > DETOXIFY_THRESHOLD or len(triggered) > 0) else 0
    except Exception as exc:
        print(f"  [warn] predict() failed: {exc}")
        return 0


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_dataset(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    dataset_name: str,
) -> dict:
    """
    Run predictions on an entire dataframe and print metrics.

    Returns a dict with keys: dataset_name, n, precision, recall, f1, tp, tn, fp, fn.
    """
    print(f"{'='*60}")
    print(f"  Dataset: {dataset_name}  ({len(df)} rows)")
    print(f"{'='*60}")

    texts  = df[text_col].fillna("").tolist()
    labels = df[label_col].astype(int).tolist()

    print(f"  Running pipeline on {len(texts)} examples...")
    preds = []
    for i, text in enumerate(texts):
        preds.append(predict(text))
        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(texts)} done...")

    # --- Metrics ---
    precision = precision_score(labels, preds, zero_division=0)
    recall    = recall_score(labels, preds, zero_division=0)
    f1        = f1_score(labels, preds, zero_division=0)
    cm        = confusion_matrix(labels, preds, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Results:")
    print(f"    Precision : {precision:.3f}")
    print(f"    Recall    : {recall:.3f}")
    print(f"    F1        : {f1:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                  CLEAN   FLAGGED")
    print(f"    Actual CLEAN    {tn:4d}    {fp:4d}   (TN / FP)")
    print(f"    Actual FLAGGED  {fn:4d}    {tp:4d}   (FN / TP)")

    # --- False Positives (predicted FLAGGED, actually CLEAN) ---
    fp_indices = [i for i, (l, p) in enumerate(zip(labels, preds)) if l == 0 and p == 1]
    print(f"\n  3 Example False Positives (flagged but actually clean):")
    if fp_indices:
        for idx in fp_indices[:3]:
            text_snippet = str(texts[idx])[:120].replace('\n', ' ')
            dtox = float(_model.predict(texts[idx])["toxicity"])
            rules_hit = run_rules(texts[idx], detoxify_toxicity=dtox)
            rule_names = [r["rule"] for r in rules_hit] or ["detoxify"]
            print(f"    [{idx}] \"{text_snippet}\"")
            print(f"          toxicity={dtox:.3f}  rules={rule_names}")
    else:
        print("    (none)")

    # --- False Negatives (predicted CLEAN, actually FLAGGED) ---
    fn_indices = [i for i, (l, p) in enumerate(zip(labels, preds)) if l == 1 and p == 0]
    print(f"\n  3 Example False Negatives (missed — actually toxic):")
    if fn_indices:
        for idx in fn_indices[:3]:
            text_snippet = str(texts[idx])[:120].replace('\n', ' ')
            dtox = float(_model.predict(texts[idx])["toxicity"])
            print(f"    [{idx}] \"{text_snippet}\"")
            print(f"          toxicity={dtox:.3f}  (below threshold, no rules hit)")
    else:
        print("    (none)")

    print()
    return {
        "dataset_name": dataset_name,
        "n": len(df),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ---------------------------------------------------------------------------
# Cross-domain comparison table
# ---------------------------------------------------------------------------

def print_comparison(results: list[dict]):
    print(f"{'='*60}")
    print("  Cross-Domain Comparison")
    print(f"{'='*60}")
    print(f"  {'Dataset':<30} {'N':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*54}")
    for r in results:
        print(
            f"  {r['dataset_name']:<30} {r['n']:>5}  "
            f"{r['precision']:>6.3f}  {r['recall']:>6.3f}  {r['f1']:>6.3f}"
        )
    print()

    if len(results) >= 2:
        sorted_r = sorted(results, key=lambda x: x["f1"], reverse=True)
        higher, lower = sorted_r[0], sorted_r[-1]
        delta = higher["f1"] - lower["f1"]
        print(f"  Key Finding:")
        print(f"    Best F1:  {higher['dataset_name']} ({higher['f1']:.3f})")
        print(f"    Worst F1: {lower['dataset_name']} ({lower['f1']:.3f})")
        print(f"    Gap: {delta:.3f}")
        if delta > 0.05:
            print(
                f"    The {delta:.3f} F1 gap suggests the pipeline generalizes unevenly\n"
                f"    across domains — likely because Detoxify and the keyword rules\n"
                f"    were calibrated on comment-style data, not conversational text.\n"
                f"    See README — Limitations section — for discussion."
            )
        else:
            print(
                f"    Performance is relatively consistent across domains,\n"
                f"    suggesting the rules + Detoxify combination generalizes well."
            )
    print()


# ---------------------------------------------------------------------------
# Main — standard evaluation (Jigsaw + Toxic Chat)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    use_wildchat = "--wildchat" in sys.argv

    jigsaw_path = os.path.join(DATA_DIR, "jigsaw_sample.csv")
    tchat_path  = os.path.join(DATA_DIR, "toxicchat_sample.csv")
    wc_path     = os.path.join(DATA_DIR, "wildchat_sample.csv")

    required = [jigsaw_path, tchat_path]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        print("Missing data files:")
        for p in missing:
            print(f"  {p}")
        print("\nRun this first:\n  python data/prepare_data.py")
        sys.exit(1)

    if use_wildchat and not os.path.exists(wc_path):
        print(f"WildChat sample not found at {wc_path}")
        print("Run this first:\n  python data/prepare_data.py")
        sys.exit(1)

    print("Loading datasets...")
    jigsaw_df = pd.read_csv(jigsaw_path)
    tchat_df  = pd.read_csv(tchat_path)
    print(f"  Jigsaw:     {len(jigsaw_df)} rows")
    print(f"  Toxic Chat: {len(tchat_df)} rows")

    results = []
    results.append(evaluate_dataset(
        df=jigsaw_df,
        text_col="comment_text",
        label_col="toxic",
        dataset_name="Jigsaw Toxicity",
    ))
    results.append(evaluate_dataset(
        df=tchat_df,
        text_col="user_input",
        label_col="toxicity",
        dataset_name="Toxic Chat",
    ))

    if use_wildchat:
        wc_df = pd.read_csv(wc_path)
        print(f"  WildChat:   {len(wc_df)} rows")
        results.append(evaluate_dataset(
            df=wc_df,
            text_col="comment_text",
            label_col="toxic",
            dataset_name="WildChat",
        ))

    print_comparison(results)
