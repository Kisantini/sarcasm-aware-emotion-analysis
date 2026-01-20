import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F
import pandas as pd
import plotly.express as px
import os
import sys

# Allow imports from src
sys.path.append(os.path.abspath("."))

from src.preprocessing.text_cleaning import clean_text

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Sarcasm-Aware Emotion Analysis",
    layout="wide",
    page_icon="🧠"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    sarcasm_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sarcasm_model = BertForSequenceClassification.from_pretrained(
        "models_saved/sarcasm_bert"
    ).to(DEVICE)
    sarcasm_model.eval()

    emotion_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    emotion_model = BertForSequenceClassification.from_pretrained(
        "models_saved/emotion_bert_7class"
    ).to(DEVICE)
    emotion_model.eval()

    return sarcasm_tokenizer, sarcasm_model, emotion_tokenizer, emotion_model

sarcasm_tokenizer, sarcasm_model, emotion_tokenizer, emotion_model = load_models()

# -----------------------------
# EMOTION LABELS & EMOJIS
# -----------------------------
EMOTION_LABELS = [
    "anger",
    "confusion",
    "disappointment",
    "joy",
    "love",
    "sadness",
    "surprise"
]


EMOTION_EMOJIS = {
    "joy": "😄", "anger": "😡", "sadness": "😢", "disappointment": "😞",
    "confusion": "🤔", "surprise": "😲"
}


# -----------------------------
# INFERENCE FUNCTIONS
# -----------------------------
def predict_sarcasm(text):
    inputs = sarcasm_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = sarcasm_model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
    return "sarcastic" if pred == 1 else "non_sarcastic"

def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_index = probs.argmax()
    prob_dict = {EMOTION_LABELS[i]: float(probs[i]) for i in range(len(EMOTION_LABELS))}
    return EMOTION_LABELS[pred_index], prob_dict

def sarcasm_aware_adjustment(emotion, sarcasm_label):
    if sarcasm_label == "sarcastic" and emotion in ["joy", "love", "surprise"]:
        return "disappointment"
    return emotion


def plot_emotion_probs(probs: dict):
    df = pd.DataFrame({"Emotion": list(probs.keys()), "Probability": list(probs.values())})
    fig = px.bar(df, x="Emotion", y="Probability", text_auto=".2f", color="Emotion", title="Emotion Probability Distribution")
    fig.update_layout(showlegend=False)
    return fig

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.title("🧠 Sarcasm-Aware Emotion AI")
    st.markdown("""
    Academic-grade NLP system for emotion analysis in customer reviews with sarcasm awareness.

    **Models Used:**
    - BERT Sarcasm Classifier
    - BERT Emotion Classifier

    **Supported Domains:** E-commerce, Products, Clothing, Electronics, Restaurants
    """)
    st.markdown("### 🎭 Supported Emotions")
    for e, emoji in EMOTION_EMOJIS.items():
        st.write(f"{emoji} {e}")

    if st.button("🔄 Reset Session"):
        st.session_state.clear()
        st.experimental_rerun()

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🎯 Sarcasm-Aware Emotion Analysis System")
st.markdown("---")

tabs = st.tabs(["📝 Single Review Analysis", "📂 Bulk Review Analysis"])

# -----------------------------
# TAB 1: SINGLE REVIEW
# -----------------------------
with tabs[0]:
    st.subheader("Single Review Analysis")
    review_text = st.text_area(
        "Enter customer review",
        height=150,
        placeholder="e.g. This phone battery lasts forever... if forever means 2 hours."
    )

    if st.button("🔍 Analyze Review"):
        if not review_text.strip():
            st.warning("Please enter a review.")
        else:
            cleaned = clean_text(review_text)
            sarcasm_pred = predict_sarcasm(cleaned)
            base_emotion, emotion_probs = predict_emotion(cleaned)
            final_emotion = sarcasm_aware_adjustment(base_emotion, sarcasm_pred)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sarcasm Detected", "YES ❗" if sarcasm_pred == "sarcastic" else "NO")
            with col2:
                st.metric("Raw Emotion", f"{EMOTION_EMOJIS.get(base_emotion, '')} {base_emotion}")
            with col3:
                st.metric("Top Confidence", f"{max(emotion_probs.values())*100:.2f}%")

            st.markdown("### 📊 Emotion Probability Distribution")
            st.plotly_chart(plot_emotion_probs(emotion_probs), use_container_width=True)

            st.markdown("### 🧠 Explanation")
            st.write(
                f"- Initial emotion: **{base_emotion}**\n"
                f"- Sarcasm detected: **{sarcasm_pred}**\n"
                f"- Adjustment applied: {'Yes' if base_emotion != final_emotion else 'No'}\n"
                f"- Final emotion: **{final_emotion}**"
            )

# -----------------------------
# TAB 2: BULK ANALYSIS
# -----------------------------
with tabs[1]:
    st.subheader("Bulk Review Analysis")
    uploaded_file = st.file_uploader("Upload CSV, Excel, TXT, or JSON", type=["csv", "xlsx", "txt", "json"])

    if uploaded_file:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith(".txt"):
                df = pd.DataFrame({"review_text": uploaded_file.read().decode().splitlines()})
            elif uploaded_file.name.endswith(".json"):
                df = pd.read_json(uploaded_file)

            st.success(f"Loaded {len(df)} reviews")

            text_column = st.selectbox("Select review text column", df.columns)

            if st.button("🚀 Run Bulk Analysis"):
                with st.spinner("Analyzing reviews..."):
                    results = []
                    for txt in df[text_column]:
                        if pd.isna(txt):
                            continue
                        cleaned = clean_text(str(txt))
                        sarcasm_pred = predict_sarcasm(cleaned)
                        base_emotion, _ = predict_emotion(cleaned)
                        final_emotion = sarcasm_aware_adjustment(base_emotion, sarcasm_pred)
                        results.append({
                            "review_text": txt,
                            "sarcasm": sarcasm_pred,
                            "raw_emotion": base_emotion,
                            "final_emotion": final_emotion
                        })

                    result_df = pd.DataFrame(results)

                st.markdown("### 📊 Emotion Distribution")
                emotion_dist = result_df["final_emotion"].value_counts()
                fig = px.pie(
                    names=emotion_dist.index,
                    values=emotion_dist.values,
                    title="Emotion Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

                sarcasm_rate = (result_df["sarcasm"] == "sarcastic").mean() * 100
                st.metric("Sarcasm Percentage", f"{sarcasm_rate:.2f}%")

                st.markdown("### Detailed Results")
                styled_df = result_df.style.apply(
                    lambda row: ["background-color: #ffe6e6" if row.sarcasm == "sarcastic" else "" for _ in row],
                    axis=1
                )
                st.dataframe(styled_df, use_container_width=True)

                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results as CSV",
                    csv,
                    "sarcasm_emotion_results.csv",
                    "text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")
