import pandas as pd
import ast
import os

TARGET_FILE = 'datasets/mimic_test.csv'
OUTPUT_FILE = 'datasets/mimic_test.csv'

def rework_dataset(path):
    if not os.path.exists(path): 
        print(f"❌ File not found: {path}")
        return
    
    df = pd.read_csv(path)

    def get_available_views(row):
        views = []
        for col in ['AP', 'PA', 'Lateral']:
            val = row.get(col)
            # Ensure the cell isn't NaN, isn't an empty string, and isn't just "[]"
            if pd.notna(val):
                clean_val = str(val).strip()
                if clean_val != "" and clean_val != "[]":
                    views.append(col)
        
        # If nothing found in specific columns, check 'image' (MIMIC style fallback)
        if not views:
            img_val = row.get('image')
            if pd.notna(img_val) and str(img_val).strip() not in ["", "[]"]:
                views.append('image')

        return str(views)

    print("Scrubbing 'Frontal' ambiguity and building view map...")
    # This replaces the old 'view' (which likely just said 'Frontal') with the list of keys
    df['view'] = df.apply(get_available_views, axis=1)

    # Drop rows that have zero images
    df_cleaned = df[df['view'] != "[]"].copy()

    # --- Label Sanitization (The [[], [], []] fix) ---
    def sanitize_labels(val):
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list) and len(parsed) == 3:
                return str(parsed)
        except: 
            pass
        return "[[], [], []]"
    
    #df_cleaned['text_label'] = df_cleaned['text_label'].apply(sanitize_labels)

    df_cleaned.to_csv(OUTPUT_FILE, index=False)
    
    print("-" * 30)
    print(f"✅ Rework Complete")
    print(f"📊 Original Rows: {len(df)}")
    print(f"📊 Final Rows:    {len(df_cleaned)}")
    print(f"📝 Sample 'view' list: {df_cleaned['view'].iloc[0]}")
    print("-" * 30)

rework_dataset(TARGET_FILE)