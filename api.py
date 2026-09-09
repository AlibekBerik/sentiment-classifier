from fastapi import FastAPI
import requests
import os

app = FastAPI(title="Sentiment Analysis API")

HF_API_URL = "https://api-inference.huggingface.co/models/Zolotouly/sentiment-classifier-model"
HF_TOKEN = os.environ.get("HF_TOKEN", "")

label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running. Use POST /predict with a 'text' field."}

@app.post("/predict")
def predict(text: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
    response = requests.post(HF_API_URL, headers=headers, json={"inputs": text})
    result = response.json()

    if isinstance(result, list) and len(result) > 0:
        top = result[0][0] if isinstance(result[0], list) else result[0]
        label = label_map.get(top["label"], top["label"])
        return {"text": text, "label": label, "confidence": round(top["score"], 4)}

    return {"text": text, "error": result}