from fastapi import FastAPI
from transformers import pipeline

app = FastAPI(title="Sentiment Analysis API")

classifier = pipeline("sentiment-analysis", model="Zolotouly/sentiment-classifier-model")
label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running. Use POST /predict with a 'text' field."}

@app.post("/predict")
def predict(text: str):
    result = classifier(text)[0]
    return {
        "text": text,
        "label": label_map.get(result["label"], result["label"]),
        "confidence": round(result["score"], 4)
    }