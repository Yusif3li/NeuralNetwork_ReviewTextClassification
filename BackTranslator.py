import pandas as pd
import time
import os
from deep_translator import GoogleTranslator
from tqdm import tqdm

INPUT_FILE = 'Dataset/train_fixed.csv' 
OUTPUT_FILE = 'Dataset/train_back_translated.csv'

def back_translate(text, target_lang='fr'):
    try:
        translator_to = GoogleTranslator(source='en', target=target_lang)
        translated = translator_to.translate(text)
        
        translator_back = GoogleTranslator(source=target_lang, target='en')
        back_translated = translator_back.translate(translated)
        return back_translated
    except:
        return text

def run_back_translation():
    print("\n" + "="*50)
    print(">>> PIPELINE STEP 2: DATA GENERATION (BACK-TRANSLATION)")
    print("="*50)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Step 1 (Cleanup) must run first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    target_classes = ['Bad', 'Very bad', 'Good']
    df_to_augment = df[df['review'].isin(target_classes)].copy()
    
    print(f">>> Rows to translate: {len(df_to_augment)}")
    print(">>> Starting Translation...")
    
    new_rows = []
    for index, row in tqdm(df_to_augment.iterrows(), total=len(df_to_augment)):
        row_id = row['id']
        original_text = str(row['clean_text'])
        label = row['review']
        
        if not original_text or original_text == "nan": 
            continue
        
        new_text = back_translate(original_text)
        
        if new_text and new_text != original_text:
            new_rows.append({'id': row_id, 'text': new_text, 'review': label})
            
        time.sleep(0.1) 

    df_new = pd.DataFrame(new_rows)
    df_new.to_csv(OUTPUT_FILE, index=False)
    print(f"Generation complete. Created {len(df_new)} new samples.\n")

if __name__ == "__main__":
    run_back_translation()