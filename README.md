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

## Uploading the Project to GitHub

### Recommended: Git from the terminal

1. Create a new empty repository on [GitHub](https://github.com/new). Do not add another README if this README is already in the project folder.
2. Open PowerShell in the project folder:

   ```powershell
   cd "C:\Users\memo\Desktop\Sentiment Analysis"
   ```

3. Initialize Git and make the first commit:

   ```powershell
   git init
   git add README.md SentimentScope_90_accuracy.ipynb
   git commit -m "Add IMDb sentiment analysis notebook"
   ```

4. Connect the local folder to the GitHub repository. Replace `YOUR-USERNAME` and `YOUR-REPOSITORY`:

   ```powershell
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
   git push -u origin main
   ```

5. Refresh the GitHub repository page and confirm that the README and notebook are visible.

### Uploading through the GitHub website

1. Create a new repository on GitHub.
2. Select **Add file** and then **Upload files**.
3. Upload `README.md` and `SentimentScope_90_accuracy.ipynb`.
4. Add a commit message such as `Add sentiment analysis notebook`.
5. Select **Commit changes**.

The website upload method is suitable for the README and notebook, but not for the large dataset archive or checkpoint.

## Large Files and GitHub Limits

Avoid committing these files to normal Git history:

- `aclImdb_v1.tar.gz` is about 84 MB.
- `sentiment_model_checkpoint.pt` is about 268 MB and exceeds GitHub's 100 MB per-file limit.
- `sentiment_model_Mehmet_Er.zip` is also a large model archive.

For a clean public repository, upload the notebook and README only, and explain that the IMDb dataset must be downloaded separately. If the trained checkpoint must be versioned, install and configure [Git LFS](https://git-lfs.com/):

```powershell
git lfs install
git lfs track "*.pt"
git add .gitattributes sentiment_model_checkpoint.pt
git commit -m "Track model checkpoint with Git LFS"
git push
```

Check the dataset license and hosting requirements before redistributing the IMDb data. Never commit passwords, API keys, or other private credentials.

## Results

The notebook prints the actual validation and test scores from the current run. Results may vary with hardware, package versions, random initialization, and training time. The from-scratch model provides a baseline; fine-tuning `distilbert-base-uncased` is intended to provide the stronger result.

## License and Dataset Notice

This repository contains educational code. The IMDb dataset is provided by the Stanford AI Lab and is subject to its own terms of use. Review the dataset's original documentation before sharing it publicly.