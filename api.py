from fastapi import FastAPI
from huggingface_hub import InferenceClient
import os

app = FastAPI(title="Sentiment Analysis API")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
client = InferenceClient(token=HF_TOKEN)

label_map = {"LABEL_0": "negative", "LABEL_1": "positive"}

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running. Use POST /predict with a 'text' field."}

@app.post("/predict")
def predict(text: str):
    try:
        result = client.text_classification(text, model="Zolotouly/sentiment-classifier-model")
        top = result[0] if isinstance(result, list) else result
        label = label_map.get(top.label, top.label)
        return {"text": text, "label": label, "confidence": round(top.score, 4)}
    except Exception as e:
        return {"text": text, "error_type": type(e).__name__, "error_message": str(e) or "no message", "error_repr": repr(e)}