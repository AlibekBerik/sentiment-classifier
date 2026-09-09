from transformers import pipeline

classifier = pipeline("sentiment-analysis", model="./sentiment_model", tokenizer="./sentiment_model")

test_reviews = [
    "This movie was absolutely fantastic, I loved every minute of it!",
    "Terrible film, complete waste of time and money.",
    "It was okay, not great but not bad either."
]

for review in test_reviews:
    result = classifier(review)
    print(f"Review: {review}")
    print(f"Prediction: {result}\n")