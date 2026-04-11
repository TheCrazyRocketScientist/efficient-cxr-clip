import os
import pandas as pd
import numpy as np
import ast
from transformers import AutoTokenizer
from tqdm import tqdm

def analyze_mimic_tokenization_for_paper_models(data_dir, tokenizer_names, sample_size=50000):
    target_files = ['mimic_train.csv', 'mimic_valid.csv', 'mimic_test.csv']
    files_to_load = [os.path.join(data_dir, f) for f in target_files if os.path.exists(os.path.join(data_dir, f))]
    
    if not files_to_load:
        print(f"Error: Could not find files in {data_dir}.")
        return

    print(f"Loading and Parsing MIMIC files...")
    all_texts = []
    
    for f in files_to_load:
        df = pd.read_csv(f)
        for col in ['text', 'text_augment']:
            if col in df.columns:
                raw_data = df[col].dropna().tolist()
                for item in raw_data:
                    try:
                        parsed_list = ast.literal_eval(item)
                        # Join the list into one clinical report string
                        full_string = " ".join([str(i) for i in parsed_list if i])
                        if full_string.strip():
                            all_texts.append(full_string)
                    except (ValueError, SyntaxError):
                        all_texts.append(str(item))

    if not all_texts:
        print("ERROR: No text data found.")
        return

    if sample_size < len(all_texts):
        texts_to_analyze = np.random.choice(all_texts, sample_size, replace=False).tolist()
    else:
        texts_to_analyze = all_texts

    results = []

    for model_id in tokenizer_names:
        print(f"Analyzing: {model_id}")
        try:
            # Force use_fast=True to ensure we get the efficient Rust-based tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
            
            # Using the direct __call__ method instead of batch_encode_plus
            # This is the most universal way to tokenize in the library
            encodings = tokenizer(
                texts_to_analyze, 
                add_special_tokens=True, 
                truncation=False, 
                padding=False
            )['input_ids']
            
            lengths = [len(ids) for ids in encodings]
            results.append({
                'Model': model_id.split('/')[-1],
                'Mean': np.mean(lengths),
                'P95': int(np.percentile(lengths, 95)),
                'P99': int(np.percentile(lengths, 99)),
                'Max': np.max(lengths)
            })
        except Exception as e:
            print(f"  - Failed {model_id}: {e}")

    if not results:
        return

    report_df = pd.DataFrame(results)
    print("\n" + "="*75)
    print("MIMIC-CXR TOKEN DISTRIBUTION REPORT")
    print("="*75)
    print(report_df.to_string(index=False))
    
    global_p99 = report_df['P99'].max()
    suggested_len = int(np.ceil(global_p99 / 32) * 32) 
    print("="*75)
    print(f"Recommended 'Golden' max_length: {suggested_len}")

# --- CONFIG ---
models_to_test = [
    'nlpie/bio-mobilebert',
    'nlpie/bio-tinybert',
    'nlpie/compact-biobert',
    'nlpie/tiny-clinicalbert'
]

analyze_mimic_tokenization_for_paper_models('/tmp/efficient-cxr-clip/datasets', models_to_test)