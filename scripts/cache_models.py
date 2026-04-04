import timm
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
from grab_correct_pr import get_first_safetensors_pr

"""
Make sure to source scripts/set_cache.sh to ensure cache is in correct location
"""


VISION_VARIANTS = [
    "timm/fastvit_t8.apple_in1k",
    "timm/fastvit_s12.apple_in1k",
    "timm/fastvit_sa24.apple_in1k",
    "timm/fastvit_ma36.apple_in1k"
]

TEXT_VARIANTS = [
    "nlpie/compact-biobert",
    "nlpie/tiny-biobert",
    "nlpie/tiny-clinicalbert",
    "nlpie/distil-biobert",
    "nlpie/bio-tinybert",
    "nlpie/distil-clinicalbert",
    "nlpie/bio-mobilebert"
]

def populate_cache():

    cache_dir = os.getenv("HF_HOME")

    for v_name in VISION_VARIANTS:
        try:
            timm.create_model(v_name, pretrained=True, num_classes=0)
        except Exception as e:
            print(f"[FAIL] {v_name}: {e}")

    for t_name in TEXT_VARIANTS:
        rev = get_first_safetensors_pr(t_name) or "main"
        try:
            """
            AutoTokenizer.from_pretrained(t_name, revision=rev)
            AutoModel.from_pretrained(t_name, revision=rev, use_safetensors=True)

            """

            snapshot_download(
                repo_id=t_name,
                revision=rev,
                cache_dir=cache_dir,
                library_name="transformers",
                # Only download safe/necessary files to save space
                allow_patterns=["*.json", "*.txt", "*.safetensors", "tokenizer.model"]
            )
        except Exception as e:
            print(f"[FAIL] {t_name}: {e}")

if __name__ == "__main__":
    populate_cache()