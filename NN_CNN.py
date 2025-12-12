import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalAveragePooling1D, Dense, Dropout, Concatenate, SpatialDropout1D, BatchNormalization, Activation)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

from Preprocessing import augment_dataset
import DataDiagnostics
import OneTimeSetup
import BackTranslator

# --- PATHS ---
TRAIN_FIXED = 'Dataset/train_fixed.csv'
TEST_FIXED = 'Dataset/test_fixed.csv'
BACK_TRANS_PATH = 'Dataset/train_back_translated.csv'
GLOVE_PATH = 'Dataset/glove.6b/glove.6B.200d.txt'

# --- SETUP ---
os.makedirs('SavedModels', exist_ok=True)
os.makedirs('ModelPredicts', exist_ok=True)
os.makedirs('new_dataset', exist_ok=True) 

# --- HYPERPARAMETERS ---
MAX_WORDS = 22000       
MAX_LEN = 300           
EMBEDDING_DIM = 200     
BATCH_SIZE = 16         
EPOCHS = 40             
TEST_SIZE = 0.2         

def check_and_prepare_data():
    """
    Ensures all data is ready before training starts.
    """
    # Check Spell Checking
    if not os.path.exists(TRAIN_FIXED) or not os.path.exists(TEST_FIXED):
        print("\n>>> Cleaned data not found. Running OneTimeSetup...")
        OneTimeSetup.run_cleanup()
    else:
        print(">>> Cleaned data found.")

    # Check Back Translation
    if not os.path.exists(BACK_TRANS_PATH):
        print("\n>>> Back-Translated data not found. Running BackTranslator...")
        print("    (This will take time, but only runs once!)")
        BackTranslator.run_back_translation()
    else:
        print(">>> Back-Translated data found.")

def load_glove_embeddings(vocab_size, word_index):
    print(f">>> Loading GloVe Embeddings from {GLOVE_PATH}...")
    if not os.path.exists(GLOVE_PATH):
        print(f"WARNING: GloVe file not found at {GLOVE_PATH}!")
        return None

    embeddings_index = {}
    try:
        with open(GLOVE_PATH, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except Exception as e:
        print(f"Error reading GloVe file: {e}")
        return None

    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    hits = 0
    misses = 0
    for word, i in word_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
    print(f"Loaded {hits} words. ({misses} words missing)")
    return embedding_matrix

def build_cnn_model(vocab_size, num_classes, input_length, embedding_matrix=None):
    inputs = Input(shape=(input_length,))
    
    if embedding_matrix is not None:
        embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(inputs)
        print("Model built with Pre-trained Embeddings (Frozen)")
    else:
        embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM)(inputs)
        print("Model built with Random Embeddings (Training from scratch)")
        
    embedding = SpatialDropout1D(0.2)(embedding)
    
    c1 = Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(0.001))(embedding)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = GlobalAveragePooling1D()(c1) 
    
    c2 = Conv1D(filters=32, kernel_size=4, padding='same', kernel_regularizer=l2(0.001))(embedding)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = GlobalAveragePooling1D()(c2) 
    
    c3 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_regularizer=l2(0.001))(embedding)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = GlobalAveragePooling1D()(c3)
    
    merged = Concatenate()([p1, p2, p3])
    
    dense = Dense(64, kernel_regularizer=l2(0.001))(merged)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dropout = Dropout(0.3)(dense)
    
    outputs = Dense(num_classes, activation='softmax')(dropout)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # PIPELINE CHECK 
    check_and_prepare_data()
    
    print("\n" + "="*50)
    print(">>> PIPELINE STEP 3: TRAINING")
    print("="*50)
    
    print(">>> Loading Data...")
    df_full = pd.read_csv(TRAIN_FIXED)
    df_full['clean_text'] = df_full['clean_text'].fillna("").astype(str)
    
    # Load Back-Translated Data
    df_back = pd.read_csv(BACK_TRANS_PATH)
    
    # DATA DIAGNOSTICS 
    print(">>> Running Data Diagnosis...")
    try:
        suggested_len = DataDiagnostics.analyze_data(df_full)
        MAX_LEN = suggested_len
        print(f">>> Updated MAX_LEN to {MAX_LEN}")
    except: pass

    print("\n>>> Splitting Train/Validation...")
    le = LabelEncoder()
    df_full['label_id'] = le.fit_transform(df_full['review'])
    classes = le.classes_
    
    # Split IDs
    indices_train, indices_val = train_test_split(
        df_full.index, 
        test_size=TEST_SIZE, 
        random_state=42, 
        stratify=df_full['label_id']
    )
    
    X_train_raw = df_full.loc[indices_train, 'clean_text']
    y_train_raw = df_full.loc[indices_train, 'review']
    X_val_raw = df_full.loc[indices_val, 'clean_text']
    y_val_raw = df_full.loc[indices_val, 'review']
    
    # MERGE BACK-TRANSLATED DATA
    if not df_back.empty:
        train_ids = df_full.loc[indices_train, 'id'].values
        # Only take rows where original ID belongs to training set
        df_back_safe = df_back[df_back['id'].isin(train_ids)].copy()
        
        if not df_back_safe.empty:
            print(f">>> Merging {len(df_back_safe)} Back-Translated samples into Training Set!")
            df_back_safe['text'] = df_back_safe['text'].astype(str)
            
            X_train_raw = pd.concat([X_train_raw, df_back_safe['text']])
            y_train_raw = pd.concat([y_train_raw, df_back_safe['review']])

    df_train_split = pd.DataFrame({'clean_text': X_train_raw, 'review': y_train_raw})

    experiments = [
        { "name": "CNN_NLTK", "strategy": "nltk", "model_file": "SavedModels/cnn_nltk.keras", "sub_file": "ModelPredicts/submission_cnn_nltk.csv", "aug_file": "new_dataset/train_aug_nltk.csv" }
    ]
    
    for exp in experiments:
        print("\n" + "#"*60)
        print(f"   STARTING PIPELINE: {exp['name']}")
        print("#"*60)
        
        # AUGMENTATION CHECK
        if os.path.exists(exp['aug_file']):
            print(f">>> Found Pre-Augmented Data: {exp['aug_file']}")
            print("    Loading directly (Skipping slow augmentation)...")
            df_train_aug = pd.read_csv(exp['aug_file'])
            df_train_aug['clean_text'] = df_train_aug['clean_text'].fillna("").astype(str)
        else:
            print(f">>> Augmenting ({exp['strategy']})...")
            df_train_aug = augment_dataset(df_train_split, strategy=exp['strategy'])
            df_train_aug.to_csv(exp['aug_file'], index=False)
        
        # TOKENIZATION
        print(">>> Tokenizing...")
        tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
        tokenizer.fit_on_texts(df_train_aug['clean_text'])
        
        # LOAD GLOVE 
        embedding_matrix = load_glove_embeddings(len(tokenizer.word_index) + 1, tokenizer.word_index)
        
        X_train_seq = tokenizer.texts_to_sequences(df_train_aug['clean_text'])
        X_val_seq = tokenizer.texts_to_sequences(X_val_raw)
        
        X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
        X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        y_train_final = le.transform(df_train_aug['review'])
        y_val_final = le.transform(y_val_raw)
        
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_final), y=y_train_final)
        weights_dict = dict(enumerate(class_weights))

        # TRAINING
        K.clear_session()
        print(f">>> Training Model: {exp['name']}...")
        model = build_cnn_model(len(tokenizer.word_index) + 1, len(classes), MAX_LEN, embedding_matrix)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ModelCheckpoint(exp['model_file'], monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
        ]
        
        model.fit(
            X_train_pad, y_train_final,
            validation_data=(X_val_pad, y_val_final),
            epochs=40, 
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=weights_dict,
            verbose=1 
        )

        # EVALUATION
        print(f"\n>>> Final Evaluation for {exp['name']}:")
        try: model.load_weights(exp['model_file']) 
        except: pass
        
        y_pred = np.argmax(model.predict(X_val_pad), axis=1)
        
        print("\n" + "="*40)
        print("CLASSIFICATION REPORT")
        print("="*40)
        print(classification_report(y_val_final, y_pred, target_names=[str(c) for c in classes], zero_division=0))
        print("CONFUSION MATRIX:")
        print(confusion_matrix(y_val_final, y_pred))
        
        # SCORE 
        acc = accuracy_score(le.inverse_transform(y_val_final), le.inverse_transform(y_pred))
        print(f"\n{'='*40}")
        print(f"OFFLINE ACCURACY SCORE: {acc:.2%}")
        print(f"{'='*40}\n")
        
        # SUBMISSION
        print(f"\n>>> Generating Submission...")
        df_test = pd.read_csv(TEST_FIXED) # Load cleaned test data
        df_test['clean_text'] = df_test['clean_text'].fillna("").astype(str)
        
        X_test_seq = tokenizer.texts_to_sequences(df_test['clean_text'])
        X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')
        
        sub_preds = np.argmax(model.predict(X_test_pad), axis=1)
        pd.DataFrame({'id': df_test['id'], 'review': le.inverse_transform(sub_preds)}).to_csv(exp['sub_file'], index=False)
        print(f"Saved: {exp['sub_file']}")

    print("\n" + "="*60)
    print("ALL MODELS COMPLETE.")
    print("="*60)