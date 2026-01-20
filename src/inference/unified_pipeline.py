import sys
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.preprocessing.text_cleaning import clean_text
from src.integration.sarcasm_aware_emotion import adjust_emotion

# ---------------------------
# Configuration
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SARCASTIC_LABELS = {0: "non_sarcastic", 1: "sarcastic"}

EMOTION_LABELS = [
    "anger",
    "confusion",
    "curiosity",
    "disappointment",
    "disgust",
    "excitement",
    "joy",
    "love",
    "optimism/approval",
    "sadness",
    "surprise"
]

# ---------------------------
# Load Models
# ---------------------------
sarcasm_tokenizer = BertTokenizer.from_pretrained("models_saved/sarcasm_bert")
sarcasm_model = BertForSequenceClassification.from_pretrained(
    "models_saved/sarcasm_bert"
).to(DEVICE)
sarcasm_model.eval()

emotion_tokenizer = BertTokenizer.from_pretrained("models_saved/emotion_bert")
emotion_model = BertForSequenceClassification.from_pretrained(
    "models_saved/emotion_bert"
).to(DEVICE)
emotion_model.eval()

# ---------------------------
# Prediction Functions
# ---------------------------
def predict_sarcasm(text):
    text = clean_text(text)
    inputs = sarcasm_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = sarcasm_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return SARCASTIC_LABELS[pred]


def predict_emotion(text):
    text = clean_text(text)
    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        outputs = emotion_model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()

    return EMOTION_LABELS[pred]


# ---------------------------
# Unified Prediction Pipeline
# ---------------------------
def analyze_review(text):
    sarcasm = predict_sarcasm(text)
    emotion = predict_emotion(text)
    final_emotion = adjust_emotion(emotion, sarcasm)

    return {
        "input_review": text,
        "sarcasm": sarcasm,
        "raw_emotion": emotion,
        "final_emotion": final_emotion
    }
