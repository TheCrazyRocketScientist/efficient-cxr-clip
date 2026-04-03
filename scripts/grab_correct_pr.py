from huggingface_hub import HfApi, list_repo_refs

def get_first_safetensors_pr(model_id):
    api = HfApi()    
    try:
        refs = list_repo_refs(model_id)
        
        for pr in refs.pull_requests:
            pr_ref = f"refs/pr/{pr.number}"
            
            files = api.list_repo_files(repo_id=model_id, revision=pr_ref)
            
            if any(f.endswith(".safetensors") for f in files):
                return pr_ref
                
        main_files = api.list_repo_files(repo_id=model_id)
        if any(f.endswith(".safetensors") for f in main_files):
            return "main"
            
    except Exception as e:
        print(f"  [ERROR] Could not access {model_id}: {e}")
    
    return None