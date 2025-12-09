import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from Preprocessing import clean_text

tqdm.pandas()

def run_cleanup():
    print("\n" + "="*50)
    print(">>> PIPELINE STEP 1: PRE-PROCESSING (SPELL CHECK)")
    print("="*50)
    
    try:
        df_train = pd.read_csv('Dataset/train.csv')
        df_test = pd.read_csv('Dataset/test.csv')
    except FileNotFoundError:
        print("Error: Could not find train.csv or test.csv in 'Dataset/' folder.")
        return

    print(">>> Cleaning TRAINING Data (Spell Check + Normalization)...")
    df_train['clean_text'] = df_train['text'].progress_apply(clean_text)
    
    print(">>> Cleaning TEST Data (Spell Check + Normalization)...")
    df_test['clean_text'] = df_test['text'].progress_apply(clean_text)
    
    print(">>> Saving Fixed Files...")
    df_train.to_csv('Dataset/train_fixed.csv', index=False)
    df_test.to_csv('Dataset/test_fixed.csv', index=False)
    
    print("PREPROCESSING COMPLETE.\n")

if __name__ == "__main__":
    run_cleanup()