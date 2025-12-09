import pandas as pd
import numpy as np
import re
import random
import nltk
from nltk.corpus import wordnet
from spellchecker import SpellChecker  

# --- SETUP NLTK ---
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# This loads a dictionary of correct English words
spell = SpellChecker()

# --- DICTIONARIES ---
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

# CLEANING FUNCTION
def clean_text(text):
    """
    Standardizes text: emojis, contractions, noise, AND SPELL CHECKING.
    """
    if pd.isna(text):
        return ""
    text = str(text)
    
    #  Emojis
    for emoji, word in EMOJI_DICT.items():
        text = text.replace(emoji, word)
    
    text = text.lower()
    
    #  Contractions
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(r"\b" + contraction + r"\b", expansion, text)
        
    # 3Repeated chars (goood -> good)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    
    #  Standard noise
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[^a-z\s]', '', text)
    
    #  SPELL CORRECTION
    # We split into words, check them, and rebuild.
    words = text.split()
    corrected_words = []
    
    # Find words that might be misspelled (unknown to the dictionary)
    misspelled = spell.unknown(words)
    
    for word in words:
        if word in misspelled:
            # Get the one most likely correct spelling
            # If correction returns None (rare), keep original
            correction = spell.correction(word)
            if correction:
                corrected_words.append(correction)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
            
    return " ".join(corrected_words).strip()

# AUGMENTATION FUNCTIONS 
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            if synonym != word and synonym.isalpha():
                synonyms.add(synonym)
    return list(synonyms)

def synonym_replacement(words, n):
    if n <= 0: return words
    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n: break
    return new_words

def random_swap(words, n):
    if n <= 0 or len(words) < 2: return words
    new_words = words.copy()
    for _ in range(n):
        idx1 = random.randint(0, len(new_words)-1)
        idx2 = idx1
        while idx1 == idx2: idx2 = random.randint(0, len(new_words)-1)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return new_words

def augment_dataset(df, strategy='basic'):
    print(f"   -> Running Augmentation Strategy: {strategy.upper()}")
    
    counts = df['review'].value_counts()
    target_count = counts.max()
    
    aug_rows = []
    
    # Group by label
    for label, group in df.groupby('review'):
        texts = group['clean_text'].tolist()
        # Add originals
        for t in texts:
            aug_rows.append({'clean_text': t, 'review': label})
        
        # Augment if needed
        current_count = len(texts)
        if current_count < target_count:
            needed = target_count - current_count
            
            generated = 0
            while generated < needed:
                txt = random.choice(texts)
                words = txt.split()
                if len(words) < 5: continue
                
                new_sent_words = words
                
                if strategy == 'basic':
                    n_change = max(1, int(0.1 * len(words)))
                    if random.random() < 0.5:
                        new_sent_words = synonym_replacement(words, n_change)
                    else:
                        new_sent_words = random_swap(words, n_change)
                        
                elif strategy == 'nltk':
                    n_change = max(1, int(0.15 * len(words)))
                    new_sent_words = synonym_replacement(words, n_change)

                new_txt = " ".join(new_sent_words)
                
                if new_txt != txt:
                    aug_rows.append({'clean_text': new_txt, 'review': label})
                    generated += 1
    
    return pd.DataFrame(aug_rows).sample(frac=1, random_state=42).reset_index(drop=True)