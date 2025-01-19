import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description='Download models from HuggingFace Hub')
    parser.add_argument('--models', nargs='+', default=[
        "TIGER-Lab/MAmmoTH2-7B",
        "Nondzu/Mistral-7B-codealpaca-lora",
        "augmxnt/shisa-gamma-7b-v1",
        "mistralai/Mistral-7B-v0.1",
        "meta-llama/Llama-2-7b",
        "TIGER-Lab/MAmmoTH-7B",
        "mrm8488/llama-2-coder-7b",
        "elyza/ELYZA-japanese-Llama-2-7b"
    ], help='List of model IDs to download')
    parser.add_argument('--cache-dir', default='/home/ubuntu/model_merging/.cache', help='Base directory for downloaded models')
    
    args = parser.parse_args()
    
    for model_id in args.models:
        local_dir = f"{args.cache_dir}/{model_id}"
        print(f"Downloading {model_id}...")
        snapshot_download(
            repo_id=model_id,
            revision="main",
            local_dir=local_dir
        )

if __name__ == '__main__':
    main()

