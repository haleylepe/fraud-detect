import pandas as pd
import os

os.makedirs("data", exist_ok=True)

df = pd.read_csv("/Users/haley3/Desktop/fraud-detect/data/jigsaw-toxic-comment-classification-challenge/train.csv")
df = df[["comment_text", "toxic"]].sample(200, random_state=42)
df.columns = ["text", "label"]
df.to_csv("data/sample.csv", index=False)

print(f"Done! Saved {len(df)} rows to data/sample.csv")