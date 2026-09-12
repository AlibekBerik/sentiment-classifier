from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()

classifier = pipeline(
    "text-classification",
    model="./sentiment_model",
    tokenizer="./sentiment_model"
)

label_map = {
    "LABEL_0": "negative",
    "LABEL_1": "positive"
}

@app.get("/")
def root():
    return {"message": "Sentiment classifier API is running"}

@app.post("/predict")
def predict(text: str):
    result = classifier(text)[0]
    return {
        "label": label_map.get(result["label"], result["label"]),
        "confidence": round(result["score"], 4)
    }