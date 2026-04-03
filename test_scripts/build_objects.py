import torch
import logging
from cxrclip.util import GlobalEnv  
from cxrclip.model import build_model
from transformers import AutoTokenizer
import traceback
logging.basicConfig(level=logging.INFO)
myobj = GlobalEnv()

def verify_models():
    # 1. Configuration for FastViT (via timm) and Tiny-BioBERT (via HF)
    model_config = {
    "name": "clip_custom",
    "image_encoder": {
        "source": "huggingface",
        "name": "timm/fastvit_t8.apple_in1k",   
        "model_type":"fastvit",
        "pretrained": True,
        "gradient_checkpointing": False,
        "cache_dir": "/home/paranjay/Desktop/research/cxr-clip/~/.cache/huggingface/hub",
    },
    "text_encoder": {
    "source": "huggingface",
    "name": "nlpie/tiny-biobert",
    "revision": "refs/pr/4", 
    "pretrained": True,
    "gradient_checkpointing": False,
    "trust_remote_code": True,
    "pooling": "mean",
    "cache_dir": "/home/paranjay/Desktop/research/cxr-clip/~/.cache/huggingface/hub",
    "local_files_only":True,
    "force_download": False, #set to true for first time in cache
    },
    "projection_head": {
        "name": "linear",
        "dropout": 0.1,
        "proj_dim": 512
    }
}
    loss_config = {"cxr_clip": {"loss_ratio": 1.0, "i2i_weight": 1.0, "t2t_weight": 0.5}}
    
    print(f"--- Step 1: Loading Tokenizer ({model_config['text_encoder']['name']}) ---")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["text_encoder"]["name"],
        cache_dir=model_config["text_encoder"]["cache_dir"]
    )

    print(f"--- Step 2: Instantiating CXR-CLIP with FastViT ---")
    try:
        model = build_model(model_config, loss_config, tokenizer=tokenizer)
        model.eval()
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"Critical Failure during instantiation: {e}")
        traceback.print_exc()
        return

    # Verify out_dim was captured correctly
    print(f"Image Encoder Feature Dim: {model.image_encoder.out_dim}")
    print(f"Text Encoder Feature Dim:  {model.text_encoder.out_dim}")

    # 3. Create dummy data
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_texts = tokenizer(["Normal lung", "Pleural effusion"], return_tensors="pt", padding=True)
    
    # We simulate the batch structure required by CXRClip.forward()
    dummy_batch = {
        "images": dummy_images,
        "image_views": dummy_images, # Simulate secondary view
        "text_tokens": dummy_texts,
        "text_tokens2": dummy_texts  # Simulate augmented text
    }

    print(f"--- Step 3: Testing Forward Pass ---")
    try:
        with torch.no_grad():
            output = model(dummy_batch)
        
        print("\n--- Verification Success! ---")
        img_emb_shape = output['image_embeddings'].shape
        txt_emb_shape = output['text_embeddings'].shape
        
        print(f"Image Embeddings: {img_emb_shape} (Expected: [{batch_size}, 512])")
        print(f"Text Embeddings:  {txt_emb_shape}  (Expected: [{batch_size}, 512])")
        
        # Check if the projection head actually worked
        if img_emb_shape[1] == 512 and txt_emb_shape[1] == 512:
            print("Projection heads correctly mapped features to shared 512-dim space.")
        else:
            print(f"Dimension mismatch: Expected 512, got {img_emb_shape[1]}")

    except Exception as e:
        print(f"\n--- Forward Pass Failure ---")
        print(f"Error: {e}")
        if "index" in str(e).lower():
            print("Action Required: Check clip.py to ensure Global Average Pooling is used for FastViT.")

if __name__ == "__main__":
    verify_models()