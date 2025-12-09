# Review Classification using Deep Learning (CNN & LSTM)

This project implements a complete Deep Learning pipeline to classify text reviews into five distinct sentiment categories: *Bad, Very Bad, Good, Very Good, Excellent*.

It addresses the challenge of a **small, imbalanced dataset** (~2,500 original samples) through rigorous data engineering, utilizing **Spell Checking**, **Back-Translation**, and **Synonym Augmentation** to synthetically expand the dataset to over 12,000 balanced samples.

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/review-classification-dl.git
cd review-classification-dl
```

### 2. Install Dependencies
Ensure Python 3.8+ is installed, then run:
```bash
pip install tensorflow pandas numpy scikit-learn nltk pyspellchecker deep-translator tqdm
```

### 3. Download GloVe Embeddings
The GloVe embedding file is too large for GitHub and must be downloaded manually.

- **Download**: [GloVe 6B (822MB)](https://nlp.stanford.edu/projects/glove/) from Stanford NLP
- **Extract**: Unzip the downloaded file
- **Place**: Move `glove.6B.100d.txt` to `Dataset/glove.6B.100d.txt`

## ⚡ Usage Guide

The project follows a 3-step sequential pipeline to handle computationally expensive preprocessing efficiently.

### Step 1: Pre-processing (Run Once)
Cleans raw text, fixes emojis, and applies spell-checking to correct typos (e.g., "awesum" → "awesome").

```bash
python OneTimeSetup.py
```
**Output**: `Dataset/train_fixed.csv`, `Dataset/test_fixed.csv`  
**Time**: ~5-10 minutes

### Step 2: Data Generation (Run Once)
Generates high-quality synthetic data by back-translating minority class reviews (English → French → English) to balance the dataset.

```bash
python BackTranslator.py
```
**Output**: `Dataset/train_back_translated.csv`  
**Time**: ~20-40 minutes (depends on internet speed)

### Step 3: Training & Ensembling (Main Experiment)
Executes the complete pipeline automatically:
- Loads cleaned & back-translated data
- Augments data using NLTK synonym replacement
- Trains CNN with frozen GloVe embeddings
- Trains Bi-LSTM with frozen GloVe embeddings
- Ensembles predictions for final results

```bash
python NN_Project.py
```

