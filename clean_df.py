import pandas as pd
import os

# Define your path
CSV_PATH = '/home/paranjay/Desktop/research/testing/efficient-cxr-clip/datasets/chexpert_train.csv'

if os.path.exists(CSV_PATH):
    print(f"Reading {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # The "Triple-List" Fix: 
    # This ensures 'positive, negative, uncertain = labels' doesn't crash.
    df['text_label'] = df['text_label'].fillna("[[], [], []]")
    
    # Optional: ensure any empty strings in the 'text' column are also handled
    if 'text' in df.columns:
        df['text'] = df['text'].fillna("")
    
    # Save it back (index=False is important to avoid adding an 'Unnamed: 0' column)
    df.to_csv(CSV_PATH, index=False)
    print("✅ CSV Sanitized with [[], [], []] structure. Ready to resume!")
else:
    print(f"❌ Error: Could not find CSV at {CSV_PATH}. Check your paths!")