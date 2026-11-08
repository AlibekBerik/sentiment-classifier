import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI

app = FastAPI()

HUB_MODEL_ID = "Zolotouly/sentiment-classifier-model"
LOCAL_MODEL_DIR = Path("./sentiment_model")

label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "positive",
}


def resolve_model_source():
    """Pick the model to load: MODEL_ID env var, then a local training run, then the Hub."""
    if os.getenv("MODEL_ID"):
        return os.environ["MODEL_ID"]
    if LOCAL_MODEL_DIR.is_dir():
        return str(LOCAL_MODEL_DIR)
    return HUB_MODEL_ID


# Loaded on first request, so the API and the tests start without the model.
@lru_cache(maxsize=1)
def get_classifier():
    from transformers import pipeline

    source = resolve_model_source()
    return pipeline("text-classification", model=source, tokenizer=source)


@app.get("/")
def root():
    return {"message": "Sentiment classifier API is running"}


@app.post("/predict")
def predict(text: str):
    result = get_classifier()(text)[0]
    return {
        "label": label_map.get(result["label"], result["label"]),
        "confidence": round(result["score"], 4),
    }
