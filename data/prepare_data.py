"""
data/prepare_data.py — Downloads and samples all evaluation datasets.

Run this once before evaluate.py:
    python data/prepare_data.py

Saves:
    data/jigsaw_sample.csv    — 200 rows  (Kaggle Jigsaw, local CSV required)
    data/toxicchat_sample.csv — 100 rows  (lmsys/toxic-chat via HuggingFace)
    data/wildchat_sample.csv  — 150 rows  (local df.csv required)

Requires: pip install datasets pandas
"""

import os
import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JIGSAW_OUT  = os.path.join(SCRIPT_DIR, "jigsaw_sample.csv")
TCHAT_OUT   = os.path.join(SCRIPT_DIR, "toxicchat_sample.csv")
WC_OUT      = os.path.join(SCRIPT_DIR, "wildchat_sample.csv")

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset 1 — Jigsaw Toxicity (local CSV from Kaggle)
# Download train.csv from:
#   https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data
# and place it at data/jigsaw-toxic-comment-classification-challenge/train.csv
# ---------------------------------------------------------------------------

def prepare_jigsaw(n: int = 200):
    local_path = os.path.join(
        SCRIPT_DIR, "jigsaw-toxic-comment-classification-challenge", "train.csv"
    )
    if not os.path.exists(local_path):
        print("  ERROR: Jigsaw train.csv not found.")
        print(f"  Expected: {local_path}")
        print("  Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data")
        return

    print("Loading Jigsaw train.csv from disk...")
    df = pd.read_csv(local_path)
    df = df[["comment_text", "toxic"]].dropna()
    df["toxic"] = df["toxic"].astype(int)

    toxic     = df[df["toxic"] == 1].sample(min(n // 2, len(df[df["toxic"] == 1])), random_state=RANDOM_SEED)
    non_toxic = df[df["toxic"] == 0].sample(min(n // 2, len(df[df["toxic"] == 0])), random_state=RANDOM_SEED)
    sample = pd.concat([toxic, non_toxic]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    sample.to_csv(JIGSAW_OUT, index=False)
    print(f"  Saved {len(sample)} rows → {JIGSAW_OUT}")
    print(f"  Toxic: {sample['toxic'].sum()} | Non-toxic: {(sample['toxic'] == 0).sum()}")


# ---------------------------------------------------------------------------
# Dataset 2 — Toxic Chat (lmsys/toxic-chat via HuggingFace)
# ---------------------------------------------------------------------------

def prepare_toxicchat(n: int = 100):
    print("Downloading lmsys/toxic-chat from HuggingFace...")
    ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train", trust_remote_code=True)
    df = ds.to_pandas()

    df = df[["user_input", "toxicity"]].dropna()
    df["toxicity"] = df["toxicity"].astype(int)

    toxic     = df[df["toxicity"] == 1]
    non_toxic = df[df["toxicity"] == 0]
    n_toxic     = min(n // 2, len(toxic))
    n_non_toxic = min(n - n_toxic, len(non_toxic))

    sample = pd.concat([
        toxic.sample(n_toxic, random_state=RANDOM_SEED),
        non_toxic.sample(n_non_toxic, random_state=RANDOM_SEED),
    ]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    sample.to_csv(TCHAT_OUT, index=False)
    print(f"  Saved {len(sample)} rows → {TCHAT_OUT}")
    print(f"  Toxic: {sample['toxicity'].sum()} | Non-toxic: {(sample['toxicity'] == 0).sum()}")


# ---------------------------------------------------------------------------
# Dataset 3 — WildChat (local df.csv)
# Place df.csv in the data/ folder. Expected columns:
#   prompt         — the text content
#   prompt_label   — "safe" or "unsafe"
# ---------------------------------------------------------------------------

def prepare_wildchat(n: int = 150):
    local_path = os.path.join(SCRIPT_DIR, "df.csv")
    if not os.path.exists(local_path):
        print(f"  Skipping WildChat — {local_path} not found.")
        print("  Place df.csv in the data/ folder to enable this dataset.")
        return

    print("Loading WildChat from df.csv...")
    df = pd.read_csv(local_path, usecols=["prompt", "prompt_label"])
    df = df[df["prompt_label"].isin(["safe", "unsafe"])].dropna(subset=["prompt"])
    df["toxic"] = (df["prompt_label"] == "unsafe").astype(int)
    df = df.rename(columns={"prompt": "comment_text"})[["comment_text", "toxic"]]
    df["comment_text"] = df["comment_text"].astype(str).str.strip()
    df = df[df["comment_text"].str.len() > 10]

    toxic     = df[df["toxic"] == 1]
    non_toxic = df[df["toxic"] == 0]
    n_toxic     = min(n // 2, len(toxic))
    n_non_toxic = min(n - n_toxic, len(non_toxic))

    sample = pd.concat([
        toxic.sample(n_toxic, random_state=RANDOM_SEED),
        non_toxic.sample(n_non_toxic, random_state=RANDOM_SEED),
    ]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    sample.to_csv(WC_OUT, index=False)
    print(f"  Saved {len(sample)} rows → {WC_OUT}")
    print(f"  Toxic: {sample['toxic'].sum()} | Non-toxic: {(sample['toxic'] == 0).sum()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_jigsaw(n=200)
    print()
    prepare_toxicchat(n=100)
    print()
    prepare_wildchat(n=150)
    print("\nAll datasets ready. Run `python evaluate.py` next.")
    print("For WildChat evaluation: python evaluate.py --wildchat")
