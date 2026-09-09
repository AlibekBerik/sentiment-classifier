\# Sentiment Classifier — Fine-Tuned DistilBERT + Deployed API



Fine-tuned a pretrained DistilBERT transformer to classify movie reviews as positive or negative, and deployed it behind a REST API for real-time inference.



\## Problem



Manually reading and categorizing large volumes of text (reviews, feedback, support tickets) doesn't scale. This project fine-tunes a pretrained language model to automate sentiment classification, and exposes it as an API any application could call.



\## Dataset



\[IMDb Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) — 50,000 labeled movie reviews (25,000 train / 25,000 test). Trained on a 3,000-review subset for practical training time; not included in this repo due to size — the dataset downloads automatically via HuggingFace's `datasets` library when the training script is run.



\## Approach



1\. \*\*Tokenization\*\* (`src/tokenize\_data.py`) — converted raw review text into numeric token IDs using DistilBERT's tokenizer, padded/truncated to a consistent 256-token length.

2\. \*\*Data preparation\*\* (`src/prepare\_data.py`) — sampled a shuffled 3,000-review training set and 1,000-review test set from the full IMDb dataset.

3\. \*\*Fine-tuning\*\* (`src/train.py`) — used transfer learning to specialize `distilbert-base-uncased` (pretrained on general text) for binary sentiment classification, via HuggingFace's `Trainer` API. Trained on a T4 GPU (Google Colab) in 2 minutes 42 seconds for 2 epochs.

4\. \*\*Evaluation\*\* — tracked accuracy and F1 per epoch; used the best-performing checkpoint (by validation loss) as the final model.

5\. \*\*Manual testing\*\* (`src/predict\_test.py`) — validated predictions on hand-written examples, including a deliberately ambiguous review, to confirm the model's confidence scores behave sensibly (high confidence on clear sentiment, lower confidence on mixed/neutral text).

6\. \*\*API deployment\*\* (`api.py`) — wrapped the fine-tuned model in a FastAPI `/predict` endpoint returning a readable label (`positive`/`negative`) and confidence score.



\## Results



| Epoch | Training Loss | Validation Loss | Accuracy | F1 |

|---|---|---|---|---|

| 1 | 0.316 | 0.334 | 85.9% | 0.866 |

| 2 | 0.234 | 0.338 | 87.3% | 0.873 |



\*\*Final model (best checkpoint): 85.9% accuracy, 0.866 F1\*\* on held-out test data.



Example predictions:

| Review | Prediction | Confidence |

|---|---|---|

| "This was the best movie I've seen all year!" | positive | 90.0% |

| "it was bad" | negative | 75.5% |

| "It was okay, not great but not bad either." | positive | 64.2% |



The model appropriately assigns lower confidence to genuinely ambiguous/mixed sentiment rather than guessing confidently — a sign of reasonable calibration, not just memorization.



\## How to run locally



```bash

python -m venv venv

venv\\Scripts\\Activate.ps1   # Windows

pip install -r requirements.txt



python src/tokenize\_data.py

python src/prepare\_data.py

python src/train.py           # or run on Colab GPU — see below

python src/predict\_test.py

uvicorn api:app --reload

```



Then visit `http://localhost:8000/docs` for the interactive API.



\*\*Note:\*\* the fine-tuned model (`sentiment\_model/`) is not included in this repo due to size (\~270MB). Running `train.py` regenerates it. For faster training, run the training step on \[Google Colab](https://colab.research.google.com) with a free T4 GPU (reduces training time from \~45+ minutes on CPU to under 3 minutes), then download the resulting model folder into the project directory before running the remaining steps locally.



\## Tech stack



Python, PyTorch, HuggingFace Transformers \& Datasets, DistilBERT, FastAPI



\## Live demo



\[Add link once deployed]

