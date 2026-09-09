from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("stanfordnlp/imdb")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Shrink to a manageable size for training on a laptop CPU
small_train = tokenized_datasets["train"].shuffle(seed=42).select(range(3000))
small_test = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

small_train.save_to_disk("data/train_dataset")
small_test.save_to_disk("data/test_dataset")

print(f"Saved {len(small_train)} training examples and {len(small_test)} test examples.")