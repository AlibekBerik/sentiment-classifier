from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")

print(dataset)
print("\nExample review:")
print(dataset["train"][0])