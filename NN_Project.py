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
from tensorflow.keras.layers import (Input, Embedding, Conv1D, GlobalAveragePooling1D, Dense, Dropout, Concatenate, SpatialDropout1D, BatchNormalization, Activation, LSTM, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K

import OneTimeSetup
import BackTranslator
import DataDiagnostics
import Preprocessing

# PATHS 
TRAIN_FIXED = 'Dataset/train_fixed.csv'
TEST_FIXED = 'Dataset/test_fixed.csv'
BACK_TRANS_PATH = 'Dataset/train_back_translated.csv'
GLOVE_PATH = 'Dataset/glove.6B.100d.txt'
AUG_FILE = 'new_dataset/train_aug_nltk.csv'

# SETUP
os.makedirs('SavedModels', exist_ok=True)
os.makedirs('ModelPredicts', exist_ok=True)
os.makedirs('new_dataset', exist_ok=True) 

# HYPERPARAMETERS 
MAX_WORDS = 22000       
MAX_LEN = 300           
EMBEDDING_DIM = 100     
BATCH_SIZE = 32        
EPOCHS = 50            
TEST_SIZE = 0.2         

def load_glove_embeddings(vocab_size, word_index):
    print(f">>> Loading GloVe Embeddings from {GLOVE_PATH}...")
    if not os.path.exists(GLOVE_PATH):
        return None

    embeddings_index = {}
    try:
        with open(GLOVE_PATH, encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
    except: return None

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

#  CNN MODEL 
def build_cnn_model(vocab_size, num_classes, input_length, embedding_matrix=None):
    inputs = Input(shape=(input_length,))
    if embedding_matrix is not None:
        embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(inputs)
    else:
        embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM)(inputs)
    
    x = SpatialDropout1D(0.2)(embedding)
    
    c1 = Conv1D(filters=32, kernel_size=3, padding='same', kernel_regularizer=l2(0.001))(x)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = GlobalAveragePooling1D()(c1) 
    
    c2 = Conv1D(filters=32, kernel_size=4, padding='same', kernel_regularizer=l2(0.001))(x)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = GlobalAveragePooling1D()(c2) 
    
    c3 = Conv1D(filters=32, kernel_size=5, padding='same', kernel_regularizer=l2(0.001))(x)
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

# LSTM MODEL 
def build_lstm_model(vocab_size, num_classes, input_length, embedding_matrix=None):
    inputs = Input(shape=(input_length,))
    
    if embedding_matrix is not None:
        embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], trainable=False)(inputs)
    else:
        embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM)(inputs)
        
    x = SpatialDropout1D(0.3)(embedding)
    
    # Bidirectional LSTM
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3, recurrent_dropout=0.0))(x)
    
    x = BatchNormalization()(x)
    x = Dense(64, kernel_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("\n" + "="*50)
    print(">>> PIPELINE: TRAINING & ENSEMBLING")
    print("="*50)
    
    #  LOAD DATA
    print(">>> Loading Data...")
    df_full = pd.read_csv(TRAIN_FIXED)
    df_full['clean_text'] = df_full['clean_text'].fillna("").astype(str)
    
    #  MERGE BACK-TRANSLATED (Using Split ID Logic)
    print(">>> Splitting IDs...")
    le = LabelEncoder()
    df_full['label_id'] = le.fit_transform(df_full['review'])
    classes = le.classes_
    
    indices_train, indices_val = train_test_split(df_full.index, test_size=TEST_SIZE, random_state=42, stratify=df_full['label_id'])
    
    X_train_raw = df_full.loc[indices_train, 'clean_text']
    y_train_raw = df_full.loc[indices_train, 'review']
    X_val_raw = df_full.loc[indices_val, 'clean_text']
    y_val_raw = df_full.loc[indices_val, 'review'] # Numeric or String? Review is string. 
    # We need the numeric Y for validation
    y_val_numeric = df_full.loc[indices_val, 'label_id'].values

    if os.path.exists(BACK_TRANS_PATH):
        print(">>> Merging Back-Translation...")
        df_back = pd.read_csv(BACK_TRANS_PATH)
        train_ids = df_full.loc[indices_train, 'id'].values
        df_back_safe = df_back[df_back['id'].isin(train_ids)].copy()
        if not df_back_safe.empty:
            df_back_safe['text'] = df_back_safe['text'].astype(str)
            X_train_raw = pd.concat([X_train_raw, df_back_safe['text']])
            y_train_raw = pd.concat([y_train_raw, df_back_safe['review']])

    #  AUGMENTATION
    if os.path.exists(AUG_FILE):
        print(f">>> Loading Augmented Data: {AUG_FILE}")
        df_train_aug = pd.read_csv(AUG_FILE)
        df_train_aug['clean_text'] = df_train_aug['clean_text'].fillna("").astype(str)
    else:
        print(">>> Generating Augmented Data...")
        df_train_split = pd.DataFrame({'clean_text': X_train_raw, 'review': y_train_raw})
        df_train_aug = Preprocessing.augment_dataset(df_train_split, strategy='nltk')
        df_train_aug.to_csv(AUG_FILE, index=False)

    #  TOKENIZATION
    print(">>> Tokenizing...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(df_train_aug['clean_text'])
    embedding_matrix = load_glove_embeddings(len(tokenizer.word_index) + 1, tokenizer.word_index)
    
    X_train_seq = tokenizer.texts_to_sequences(df_train_aug['clean_text'])
    X_val_seq = tokenizer.texts_to_sequences(X_val_raw)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    
    y_train_final = le.transform(df_train_aug['review'])
    
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_final), y=y_train_final)
    weights_dict = dict(enumerate(class_weights))

    #  EXPERIMENTS 
    experiments = [
        { 
            "type": "CNN",
            "name": "CNN_Model", 
            "model_file": "SavedModels/cnn_best.keras", 
            "sub_file": "ModelPredicts/submission_cnn_best.csv", 
        },
        { 
            "type": "LSTM",
            "name": "LSTM_Model", 
            "model_file": "SavedModels/lstm_best.keras", 
            "sub_file": "ModelPredicts/submission_lstm_best.csv", 
        }
    ]
    
    model_predictions_val = [] # For offline ensemble accuracy
    model_predictions_test = [] # For submission
    
    # Load Test Data for prediction
    df_test = pd.read_csv(TEST_FIXED)
    df_test['clean_text'] = df_test['clean_text'].fillna("").astype(str)
    X_test_seq = tokenizer.texts_to_sequences(df_test['clean_text'])
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post')

    for exp in experiments:
        print("\n" + "#"*60)
        print(f"   TRAINING: {exp['name']} ({exp['type']})")
        print("#"*60)
        
        K.clear_session()
        
        if exp['type'] == 'CNN':
            model = build_cnn_model(len(tokenizer.word_index) + 1, len(classes), MAX_LEN, embedding_matrix)
        elif exp['type'] == 'LSTM':
            model = build_lstm_model(len(tokenizer.word_index) + 1, len(classes), MAX_LEN, embedding_matrix)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
            ModelCheckpoint(exp['model_file'], monitor='val_accuracy', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
        ]
        
        model.fit(
            X_train_pad, y_train_final,
            validation_data=(X_val_pad, y_val_numeric),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=weights_dict,
            verbose=1 
        )
        
        # Validation Predictions
        print(f"\n>>> Evaluating {exp['name']}...")
        val_probs = model.predict(X_val_pad)
        val_preds = np.argmax(val_probs, axis=1)
        acc = accuracy_score(y_val_numeric, val_preds)
        print(f"    Accuracy: {acc:.2%}")
        
        # Store probs for ensemble
        model_predictions_val.append(val_probs)
        
        # Test Predictions
        test_probs = model.predict(X_test_pad)
        model_predictions_test.append(test_probs)
        
        # Save Individual Submission
        sub_preds = np.argmax(test_probs, axis=1)
        pd.DataFrame({'id': df_test['id'], 'review': le.inverse_transform(sub_preds)}).to_csv(exp['sub_file'], index=False)

    #  ENSEMBLING 
    print("\n" + "="*50)
    print(">>> ENSEMBLING (CNN + LSTM)")
    print("="*50)
    
    #  Validation Score
    avg_probs_val = (model_predictions_val[0] + model_predictions_val[1]) / 2
    final_preds_val = np.argmax(avg_probs_val, axis=1)
    ensemble_acc = accuracy_score(y_val_numeric, final_preds_val)
    
    print(f"OFFLINE ENSEMBLE ACCURACY: {ensemble_acc:.2%}")
    
    #  Test Submission
    avg_probs_test = (model_predictions_test[0] + model_predictions_test[1]) / 2
    final_preds_test = np.argmax(avg_probs_test, axis=1)
    
    sub_file = "ModelPredicts/submission_ensemble.csv"
    pd.DataFrame({
        'id': df_test['id'], 
        'review': le.inverse_transform(final_preds_test)
    }).to_csv(sub_file, index=False)
    
    print(f"Ensemble Submission Saved: {sub_file}")
    print("Done!")