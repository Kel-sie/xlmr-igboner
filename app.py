# app.py
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Load your trained model + tokenizer from HF
MODEL_NAME = "preshnkelsie/xlmr-igboner"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)

# Your BIO label mapping
id2label = {
    0: "B-DATE",
    1: "B-LOC",
    2: "B-ORG",
    3: "B-PER",
    4: "I-DATE",
    5: "I-LOC",
    6: "I-ORG",
    7: "I-PER",
    8: "O"
}

# Colors for entities
ENTITY_COLORS = {
    "B-PER": "#FFB6C1", "I-PER": "#FF69B4",
    "B-LOC": "#ADD8E6", "I-LOC": "#87CEEB",
    "B-ORG": "#90EE90", "I-ORG": "#32CD32",
    "B-DATE": "#FFD580", "I-DATE": "#FFB347",
    "O": "#F0F0F0"
}

def predict(text):
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
    with torch.no_grad():
        logits = model(**tokens).logits
    
    # Get predictions
    predictions = torch.argmax(logits, dim=2).squeeze().tolist()
    token_ids = tokens["input_ids"].squeeze().tolist()
    tokens_decoded = tokenizer.convert_ids_to_tokens(token_ids)

    # Align labels with tokens
    results = []
    for token, pred_id in zip(tokens_decoded, predictions):
        label = id2label[pred_id]
        if token not in ["[CLS]", "[SEP]", "<s>", "</s>", "<pad>"]:
            results.append((token, label))
    return results

def display_colored(tokens_labels):
    """Display tokens with background colors by label."""
    html = ""
    for token, label in tokens_labels:
        color = ENTITY_COLORS.get(label, "#FFFFFF")
        html += f"<span style='background-color:{color}; padding:3px; margin:2px; border-radius:4px'>{token} ({label})</span> "
    st.markdown(html, unsafe_allow_html=True)

# Streamlit UI
st.title("🌍 Igbo NER with XLM-R (BIO Tags)")
st.write("This app highlights tokens with their **BIO NER tags** using our fine-tuned model.")

user_input = st.text_area("✍🏾 Enter Igbo text:", "Chika biara Owerri n’afọ 2020")

if st.button("Analyze"):
    results = predict(user_input)
    st.subheader("🔎 BIO-tagged Output")
    display_colored(results)

    # Raw output table
    st.subheader("📋 Token Predictions")
    st.table(results)
