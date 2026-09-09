from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("stanfordnlp/imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(tokenized_datasets["train"][0])