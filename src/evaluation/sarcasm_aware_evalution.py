import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.inference.unified_pipeline import predict_emotion, predict_sarcasm
from src.integration.sarcasm_aware_emotion import adjust_emotion

# ---------------------------
# Load sarcasm dataset
# ---------------------------
df = pd.read_csv("data/processed/sarcasm_clean.csv")

df = df[df["sarcasm_label"] == "sarcastic"].reset_index(drop=True)
print("Sarcastic samples:", len(df))

# ---------------------------
# Evaluation counters
# ---------------------------
positive_emotions = {
    "joy", "love", "excitement", "optimism/approval"
}

total = 0
corrected = 0

examples = []

# ---------------------------
# Run evaluation
# ---------------------------
for _, row in df.iterrows():
    text = row["review_text"]

    raw_emotion = predict_emotion(text)
    sarcasm = predict_sarcasm(text)
    final_emotion = adjust_emotion(raw_emotion, sarcasm)

    total += 1

    if raw_emotion in positive_emotions and final_emotion != raw_emotion:
        corrected += 1
        if len(examples) < 5:
            examples.append((text, raw_emotion, final_emotion))

# ---------------------------
# Results
# ---------------------------
print("\nSarcasm-Aware Adjustment Results")
print(f"Total sarcastic reviews: {total}")
print(f"Positive emotions corrected: {corrected}")
print(f"Correction rate: {corrected / total:.2%}")

print("\nExample corrections:")
for ex in examples:
    print("\nReview:", ex[0])
    print("Baseline emotion:", ex[1])
    print("Sarcasm-aware emotion:", ex[2])
