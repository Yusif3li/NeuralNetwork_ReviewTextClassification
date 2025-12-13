import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# CONFIGURATION
MAX_LEN = 300  
EXAM_DATA_PATH = 'Dataset/test_fixed.csv'  # UPDATE THIS with the data path

# PATHS FOR  CNN
PATH_CNN_ONLY_MODEL = 'SavedModels/cnn_nltk.keras'
PATH_CNN_ONLY_TOK = 'SavedModels/tokenizer_cnn.pickle'
PATH_CNN_ONLY_LE = 'SavedModels/label_encoder_cnn.pickle'

# PATHS FOR ENSEMBLE (CNN + LSTM)
PATH_ENS_CNN_MODEL = 'SavedModels/cnn_best.keras'
PATH_ENS_LSTM_MODEL = 'SavedModels/lstm_best.keras'
PATH_ENS_TOK = 'SavedModels/tokenizer_ensemble.pickle'
PATH_ENS_LE = 'SavedModels/label_encoder_ensemble.pickle'

EMOJI_DICT = {
    ":)": " happy ", ":-)": " happy ", ":D": " happy ", "xD": " happy ", 
    "😊": " happy ", "😍": " loved ", "👍": " good ", "❤": " love ",
    ":(": " sad ", ":-(": " sad ", ":'(": " crying ", "😭": " crying ",
    "😡": " angry ", "🤬": " angry ", "👎": " bad ", "😞": " sad ",
    "😩": " sad ", "😫": " sad "
}

CONTRACTIONS = {
    "don't": "do not", "can't": "cannot", "won't": "will not", 
    "i'm": "i am", "it's": "it is", "he's": "he is", "she's": "she is",
    "that's": "that is", "what's": "what is", "where's": "where is",
    "there's": "there is", "who's": "who is", "how's": "how is",
    "let's": "let us", "didn't": "did not", "couldn't": "could not",
    "wouldn't": "would not", "shouldn't": "should not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "you're": "you are", "we're": "we are", "they're": "they are",
    "you've": "you have", "we've": "we have", "they've": "they have",
    "you'll": "you will", "we'll": "we will", "they'll": "they will",
    "i've": "i have", "i'll": "i will"
}

def clean_text_fast(text):
    if pd.isna(text): 
        return ""
    text = str(text)
    
    # Emojis
    for emoji, word in EMOJI_DICT.items():
        text = text.replace(emoji, word)
    
    text = text.lower()
    
    # Contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r"\b" + contraction + r"\b", expansion, text)
        
    # Repeated chars & Noise
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
        
    return text.strip()


def run_cnn(df):
    print("\n" + "#"*60)
    print(">>> RUNNING: STANDALONE CNN EVALUATION")
    print("#"*60)
    
    # 1. Load
    try:
        print(f"Loading Tokenizer: {PATH_CNN_ONLY_TOK}")
        with open(PATH_CNN_ONLY_TOK, 'rb') as h: tokenizer = pickle.load(h)
        
        print(f"Loading Label Encoder: {PATH_CNN_ONLY_LE}")
        with open(PATH_CNN_ONLY_LE, 'rb') as h: le = pickle.load(h)
        
        print(f"Loading Model: {PATH_CNN_ONLY_MODEL}")
        model = load_model(PATH_CNN_ONLY_MODEL)
    except FileNotFoundError as e:
        print(f"SKIPPING CNN: Missing file ({e})")
        return

    # 2. Prepare Data
    print("Tokenizing data")
    X_seq = tokenizer.texts_to_sequences(df['clean_text'])
    X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict
    print("Predicting")
    y_pred = np.argmax(model.predict(X_pad), axis=1)
    labels = le.inverse_transform(y_pred)
    
    # 4. Score 
    if 'review' in df.columns:
        acc = accuracy_score(df['review'].astype(str), labels)
        print(f"\n>> CNN ACCURACY: {acc:.2%}")
        print(classification_report(df['review'].astype(str), labels, zero_division=0))
    
    # 5. Save
    outfile = 'exam_predictions_CNN.csv'
    df_out = df.copy()
    df_out['predicted_review'] = labels
    df_out.to_csv(outfile, index=False)
    print(f"Saved: {outfile}")

def run_ensemble(df):
    print("\n" + "#"*60)
    print(">>> RUNNING: ENSEMBLE (CNN + LSTM) EVALUATION")
    print("#"*60)
    
    # 1. Load 
    try:
        print(f"Loading Tokenizer: {PATH_ENS_TOK}")
        with open(PATH_ENS_TOK, 'rb') as h: tokenizer = pickle.load(h)
        
        print(f"Loading Label Encoder: {PATH_ENS_LE}")
        with open(PATH_ENS_LE, 'rb') as h: le = pickle.load(h)
        
        print(f"Loading CNN Model: {PATH_ENS_CNN_MODEL}")
        model_cnn = load_model(PATH_ENS_CNN_MODEL)
        
        print(f"Loading LSTM Model: {PATH_ENS_LSTM_MODEL}")
        model_lstm = load_model(PATH_ENS_LSTM_MODEL)
    except FileNotFoundError as e:
        print(f"SKIPPING ENSEMBLE: Missing file ({e})")
        return

    # 2. Prepare Data
    print("Tokenizing data")
    X_seq = tokenizer.texts_to_sequences(df['clean_text'])
    X_pad = pad_sequences(X_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict 
    print("Predicting (Ensemble)")
    probs_cnn = model_cnn.predict(X_pad)
    probs_lstm = model_lstm.predict(X_pad)
    avg_probs = (probs_cnn + probs_lstm) / 2
    
    y_pred = np.argmax(avg_probs, axis=1)
    labels = le.inverse_transform(y_pred)
    
    # 4. Score
    if 'review' in df.columns:
        acc = accuracy_score(df['review'].astype(str), labels)
        print(f"\n>> ENSEMBLE ACCURACY: {acc:.2%}")
        print(classification_report(df['review'].astype(str), labels, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(df['review'].astype(str), labels))

    # 5. Save
    outfile = 'exam_predictions_ensemble.csv'
    df_out = df.copy()
    df_out['predicted_review'] = labels
    df_out.to_csv(outfile, index=False)
    print(f"Saved: {outfile}")

if __name__ == "__main__":

    print(f">>> Loading Exam Data: {EXAM_DATA_PATH}")
    if not os.path.exists(EXAM_DATA_PATH):
        print("ERROR: Test data file not found!")
        exit()
        
    df = pd.read_csv(EXAM_DATA_PATH)
    
    # Handle column names
    text_col = 'text' if 'text' in df.columns else 'clean_text'
    if text_col not in df.columns:
        print("ERROR: No 'text' column found.")
        exit()

    # Preprocessing
    print(">>> Running Fast Preprocessing")
    df['clean_text'] = df[text_col].astype(str).apply(clean_text_fast)
    
    # Run sctipts
    run_cnn(df)
    run_ensemble(df)
    
    print("\n" + "="*60)
    print("ALL EVALUATIONS COMPLETE.")
    print("="*60)