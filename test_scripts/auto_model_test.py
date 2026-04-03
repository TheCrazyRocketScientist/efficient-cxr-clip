from cxrclip.util import GlobalEnv
GlobalEnv() 

# SECOND: Now import everything else
import torch
import logging
from cxrclip.model import build_model
from transformers import AutoModel

logging.basicConfig(level=logging.INFO)

def direct_debug():
    # Use the EXACT path from your screenshot
    cache_dir = "/home/paranjay/~/.cache/huggingface/hub"
    model_name = "timm/fastvit_t8.apple_in1k"
    
    print(f"Attempting direct AutoModel load from: {model_name}")
    try:
        # We bypass the cxrclip wrapper to find the true point of failure
        model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True, 
            cache_dir=cache_dir,
            local_files_only=False # Force False to ensure it checks the Hub
        )
        print("SUCCESS: AutoModel loaded directly.")
        print(f"Model type: {type(model)}")
        
        # Check output shape
        dummy_in = torch.randn(1, 3, 224, 224)
        out = model(dummy_in)
        # FastViT usually returns a dict-like object or a tensor
        if hasattr(out, 'last_hidden_state'):
            print(f"Output shape: {out.last_hidden_state.shape}")
        else:
            print(f"Output shape: {out.shape}")

    except Exception as e:
        print(f"DIRECT LOAD FAILED: {e}")

if __name__ == "__main__":
    direct_debug()