import pandas as pd
import ast
import os

CSV_PATH = rf"datasets/chexpert_valid.csv"

# Initialize our counter
cleaned_count = 0

def sanitize_and_count(val):
    global cleaned_count
    try:
        parsed = ast.literal_eval(str(val))
        # If it's a list and has exactly 3 elements, it's already "healthy"
        if isinstance(parsed, list) and len(parsed) == 3:
            return str(parsed)
    except:
        pass
    
    # If we reach here, the row was "mangled" (NaN, [], or wrong length)
    cleaned_count += 1
    return "[[], [], []]"

if os.path.exists(CSV_PATH):
    print(f"🔍 Auditing {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    
    # Run the sanitization
    df['text_label'] = df['text_label'].apply(sanitize_and_count)
    
    # Save the cleaned version
    df.to_csv(CSV_PATH, index=False)
    
    print("-" * 30)
    print(f"✅ SANITIZATION COMPLETE")
    print(f"📊 Total Rows Processed: {total_rows}")
    print(f"🧹 Mangled Rows Fixed:  {cleaned_count}")
    print(f"📈 Cleanliness Rate:    {((total_rows - cleaned_count) / total_rows) * 100:.2f}%")
    print("-" * 30)
else:
    print("❌ Error: CSV not found. Check your paths!")