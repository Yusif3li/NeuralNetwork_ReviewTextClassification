
# Review Classification with Deep Learning

A comprehensive deep learning project for sentiment classification using CNN and Bi-LSTM with GloVe embeddings.

## 🛠️ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/review-classification-dl.git
    cd review-classification-dl
    ```

2. **Install dependencies:**
    ```bash
    pip install tensorflow pandas numpy scikit-learn nltk pyspellchecker deep-translator tqdm
    ```

3. **Download GloVe embeddings:**
    - Download [GloVe 6B (822MB)](https://nlp.stanford.edu/projects/glove/) from Stanford NLP
    - Extract the zip file
    - Place `glove.6B.100d.txt` into the `Dataset/` folder
    - Verify: `Dataset/glove.6B.100d.txt`

## ⚡ Quick Start

The project runs in 3 sequential steps:

### Step 1: Preprocessing (5-10 min)
```bash
python OneTimeSetup.py
```
Cleans text, fixes emojis, and corrects typos. Outputs: `Dataset/train_fixed.csv`, `Dataset/test_fixed.csv`

### Step 2: Data Generation (20-40 min)
```bash
python BackTranslator.py
```
Generates synthetic data via back-translation. Outputs: `Dataset/train_back_translated.csv`

### Step 3: Training & Ensembling
```bash
python NN_Project.py
```
Trains CNN and Bi-LSTM models, ensembles predictions, and generates submission.

## 📂 Project Structure

```
├── Dataset/
│   ├── train.csv
│   ├── test.csv
│   └── glove.6B.100d.txt (download required)
├── SavedModels/
├── ModelPredicts/
├── NN_Project.py
├── Preprocessing.py
├── OneTimeSetup.py
├── BackTranslator.py
├── DataDiagnostics.py
└── requirements.txt
```

