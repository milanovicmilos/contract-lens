import os
from kaggle.api.kaggle_api_extended import KaggleApi

def main():
    """
    Automates the submission of the src/infrastructure/ai/kaggle_train_lora.py 
    script to Kaggle. Requires Kaggle credentials in the environment or 
    ~/.kaggle/kaggle.json.
    """
    # 1. Initialize and authenticate via environment variables
    # The API will automatically pick up KAGGLE_USERNAME and KAGGLE_API_KEY from .env 
    # if loaded or exported to the system.
    print("Authenticating with Kaggle API...")
    api = KaggleApi()
    api.authenticate()
    
    # 2. Setup Meta Data for pushing a Script
    # In order to push code to Kaggle as a notebook/script, we need a kernel-metadata.json
    kernel_metadata = {
      "id": f"{os.environ.get('KAGGLE_USERNAME', 'username')}/contractlens-lora-training",
      "title": "ContractLens DeBERTa LoRA Training",
      "code_file": "src/infrastructure/ai/kaggle_train_lora.py",
      "language": "python",
      "kernel_type": "script",
      "is_private": "true",
      "enable_gpu": "true",
      "enable_internet": "true",
      "dataset_sources": [],
      "competition_sources": [],
      "kernel_sources": []
    }
    
    import json
    with open("kernel-metadata.json", "w") as f:
        json.dump(kernel_metadata, f, indent=4)
        
    print("Pushing tracking script to Kaggle...")
    try:
        api.kernels_push(".")
        print("Successfully initiated Kaggle job! Check Kaggle dashboard for progress.")
    except Exception as e:
        print(f"Failed to push to Kaggle. Ensure API keys are correct. Error: {e}")
        
    # Cleanup metadata
    if os.path.exists("kernel-metadata.json"):
        os.remove("kernel-metadata.json")

if __name__ == "__main__":
    main()
