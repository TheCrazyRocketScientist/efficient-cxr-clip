import wandb
import os

api = wandb.Api()
project_path = "starmariner027-manipal-institute-of-technology-bangalore/CXR-CLIP" 
download_dir = "./model_checkpoints"
os.makedirs(download_dir, exist_ok=True)

artifact_type = api.artifact_type(type_name="model", project=project_path)

for collection in artifact_type.collections():
    artifact = api.artifact(f"{project_path}/{collection.name}:latest")
    
    temp_path = artifact.download()
    
    original_file = os.path.join(temp_path, "model-best.tar")
    new_file_name = f"{collection.name}.tar"
    final_path = os.path.join(download_dir, new_file_name)
    
    if os.path.exists(original_file):
        os.rename(original_file, final_path)
        print(f"Successfully saved: {new_file_name}")
