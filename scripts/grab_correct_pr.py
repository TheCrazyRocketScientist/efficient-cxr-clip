# scripts/grab_correct_pr.py

import argparse
from huggingface_hub import HfApi, list_repo_refs

def get_first_safetensors_pr(model_id):
    """
    Checks Pull Requests for the first one containing .safetensors files.
    Defaults to 'main' if safetensors exist there.
    """
    api = HfApi()    
    try:
        # Check current main branch first for efficiency
        main_files = api.list_repo_files(repo_id=model_id)
        if any(f.endswith(".safetensors") for f in main_files):
            return "main"

        # If not in main, check Pull Requests
        refs = list_repo_refs(model_id)
        
        # Pull requests are usually ordered by number; we iterate to find the first match
        for pr in refs.pull_requests:
            pr_ref = f"refs/pr/{pr.number}"
            
            try:
                files = api.list_repo_files(repo_id=model_id, revision=pr_ref)
                if any(f.endswith(".safetensors") for f in files):
                    return pr_ref
            except Exception:
                continue
                
    except Exception as e:
        print(f"  [ERROR] Could not access {model_id}: {e}")
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch the correct PR revision containing safetensors.")
    parser.add_argument("--model_id", type=str, required=True, help="The Hugging Face model ID (e.g., 'nlpie/tiny-biobert')")
    
    args = parser.parse_args()
    
    revision = get_first_safetensors_pr(args.model_id)
    
    if revision:
        print(revision)
    else:
        print("main")