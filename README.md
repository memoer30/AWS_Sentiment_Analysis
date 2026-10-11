# SentimentScope: IMDb Sentiment Analysis

SentimentScope is a PyTorch project that classifies IMDb movie reviews as **positive** or **negative**. The main notebook, [`SentimentScope_90_accuracy.ipynb`](SentimentScope_90_accuracy.ipynb), contains the complete workflow:

- Loading and exploring the IMDb dataset
- Splitting labeled reviews into training and validation sets
- Tokenizing reviews with Hugging Face `bert-base-uncased`
- Building PyTorch `Dataset` and `DataLoader` objects
- Training a transformer classifier from scratch
- Fine-tuning the pretrained `distilbert-base-uncased` model
- Evaluating validation and test accuracy
- Saving the best model and predicting sentiment for new reviews

## Project Files

```text
Sentiment Analysis/
├── SentimentScope_90_accuracy.ipynb  # Main notebook
├── aclImdb/                          # Extracted IMDb dataset
├── aclImdb_v1.tar.gz                 # Original dataset archive (optional)
├── sentiment_model_checkpoint.pt     # Saved PyTorch checkpoint (optional)
└── README.md
```

The `SentimentScope_completed.ipynb` and `SentimentScope_starter.ipynb` files are additional notebook versions. The 90%-accuracy notebook is the recommended entry point.

## Requirements

- Python 3.9 or newer
- Jupyter Notebook or VS Code with the Jupyter extension
- A CUDA-enabled GPU is strongly recommended for training
- Approximately 8 GB of available disk space for the dataset, model files, and Python packages

Install the main dependencies in a terminal or notebook cell:

```bash
pip install torch pandas numpy matplotlib seaborn transformers jupyter
```

The first use of the Hugging Face tokenizer and pretrained model requires an internet connection so the model files can be downloaded.

## Dataset Setup

The notebook expects this relative folder structure:

```text
aclImdb/
├── train/
│   ├── pos/
│   ├── neg/
│   └── unsup/
└── test/
    ├── pos/
    └── neg/
```

If only `aclImdb_v1.tar.gz` is available, extract it in the same directory as the notebook:

```bash
tar -xzf aclImdb_v1.tar.gz
```

On Windows, the archive can also be extracted with File Explorer or 7-Zip. The extracted folder must be named `aclImdb` and placed beside the notebook.

## Running the Notebook

1. Open `SentimentScope_90_accuracy.ipynb` in VS Code or Jupyter.
2. Select a Python environment with the required packages installed.
3. Run the cells from top to bottom.
4. Confirm that the dataset checks report 25,000 training reviews and 25,000 test reviews.
5. Run the from-scratch transformer section to establish the baseline.
6. Run the pretrained fine-tuning section to train `distilbert-base-uncased`.
7. Review the validation and test accuracy summary.
8. Run the final cell to save `sentiment_model_checkpoint.pt` and test the prediction interface.

The notebook uses 256-token inputs, a batch size of 32, and keeps the best validation checkpoint. Training time depends on the GPU. The pretrained model section is the recommended route for reaching the 90% test-accuracy target.

## Example Prediction

The final notebook section provides `predict_sentiment()` for one review or a list of reviews. It returns the predicted label, sentiment name, and confidence score:

```python
predict_sentiment(
    "This movie was an absolute masterpiece!",
    final_model,
    tokenizer,
    device,
)
```

Labels are encoded as:

- `0`: Negative
- `1`: Positive
