# Review Classification using Deep Learning (CNN & LSTM)

This project implements a complete Deep Learning pipeline to classify text reviews into five distinct sentiment categories (*Bad, Very Bad, Good, Very Good, Excellent*).

It tackles the challenge of a **small, imbalanced dataset** (~2,500 original samples) through rigorous data engineering, utilizing **Spell Checking**, **Back-Translation**, and **Synonym Augmentation** to synthetically expand the dataset to over 12,000 balanced samples.

## 🚀 Key Features

* **Multi-Model Architecture:** Implements a 3-Branch 1D-CNN and a Bidirectional LSTM.
* **Advanced Preprocessing:**
    * **Spell Correction:** Uses `pyspellchecker` to fix typos before tokenization.
    * **Back-Translation:** Generates new training samples by translating English $\to$ French $\to$ English.
    * **NLTK Augmentation:** Balances classes using synonym replacement.
* **Transfer Learning:** Utilizes Pre-trained **GloVe 100d** embeddings.
* **Ensembling:** Combines CNN and LSTM predictions via soft voting for improved robustness.
* **Anti-Leakage Pipeline:** Ensures data augmentation occurs *after* the train/validation split to guarantee honest evaluation scores.

## 📂 Project Structure

```text
├── Dataset/
│   ├── train.csv                # Original Training Data
│   ├── test.csv                 # Original Test Data
│   ├── glove.6B.100d.txt        # Pre-trained Embeddings (Download required)
│   └── ... (Generated csv files appear here)
├── SavedModels/                 # Trained .keras models saved here
├── ModelPredicts/               # Submission CSVs saved here
├── NN_Project.py                # MAIN SCRIPT: Trains models & generates submissions
├── Preprocessing.py             # Utility: Cleaning & Augmentation logic
├── OneTimeSetup.py              # Utility: Runs Spell Checker (Step 1)
├── BackTranslator.py            # Utility: Generates new data (Step 2)
├── DataDiagnostics.py           # Utility: Analyzes class balance & length
└── requirements.txt             # Python dependencies