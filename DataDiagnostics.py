import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter

def analyze_data(df, text_col='text', label_col='review'):

    print("\n" + "="*60)
    print(">>> PHASE 1: RAW DATA DIAGNOSIS")
    print("="*60)
    
    # Basic Stats
    print(f"Total Rows: {len(df)}")
    print(f"Missing Values:\n{df.isnull().sum()}")
    
    # Class Balance
    print("\n--- Class Distribution (Original) ---")
    dist = df[label_col].value_counts()
    print(dist)
    
    if dist.max() / dist.min() > 2:
        print(" WARNING: Severe Class Imbalance Detected!")
    
    # Length Analysis
    df['text_len'] = df[text_col].astype(str).apply(lambda x: len(x.split()))
    p95 = int(df['text_len'].quantile(0.95))
    print(f"\n--- Text Length Stats ---")
    print(f"Avg Words: {df['text_len'].mean():.1f}")
    print(f"Max Words: {df['text_len'].max()}")
    print(f"95th Percentile: {p95} words (Recommended MAX_LEN)")
    
    return p95

def inspect_processed_data(df, name="Processed Data"):
    print(f"\n>>> INSPECTING: {name}")
    print(f"Total Rows: {len(df)}")
    print(f"Class Distribution:\n{df['review'].value_counts()}")
    
    # Check for empty strings
    empty_count = df[df['clean_text'] == ""].shape[0]
    if empty_count > 0:
        print(f" WARNING: Found {empty_count} empty rows after cleaning!")
    else:
        print("No empty rows found.")