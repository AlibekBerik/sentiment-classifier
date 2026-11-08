"""Gradio demo for Hugging Face Spaces: try the fine-tuned DistilBERT in the browser."""
import gradio as gr
from transformers import pipeline

MODEL_ID = "Zolotouly/sentiment-classifier-model"
LABELS = {"LABEL_0": "negative", "LABEL_1": "positive"}

classifier = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID, top_k=None)


def predict(text):
    if not text.strip():
        return {}
    scores = classifier(text, truncation=True, max_length=256)
    # Depending on the transformers version, one input gives [{...}, {...}] or [[{...}, {...}]]
    if scores and isinstance(scores[0], list):
        scores = scores[0]
    return {LABELS.get(s["label"], s["label"]): s["score"] for s in scores}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, label="Movie review"),
    outputs=gr.Label(label="Sentiment"),
    title="Movie Review Sentiment (fine-tuned DistilBERT)",
    description=(
        "DistilBERT fine-tuned on 3,000 IMDb reviews: 85.9% accuracy, 0.866 F1. "
        "Code: https://github.com/AlibekBerik/sentiment-classifier"
    ),
    examples=[
        ["This movie was absolutely fantastic, I loved every minute of it!"],
        ["Terrible film, complete waste of time and money."],
        ["It was okay, not great but not bad either."],
    ],
)

if __name__ == "__main__":
    demo.launch()
