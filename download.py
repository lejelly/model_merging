from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TIGER-Lab/MAmmoTH2-7B", 
    revision="main",
    local_dir=".cache/TIGER-Lab/MAmmoTH2-7B"
)
snapshot_download(
    repo_id="Nondzu/Mistral-7B-codealpaca-lora", 
    revision="main",
    local_dir=".cache/Nondzu/Mistral-7B-codealpaca-lora"
)
snapshot_download(
    repo_id="augmxnt/shisa-gamma-7b-v1", 
    revision="main",
    local_dir=".cache/augmxnt/shisa-gamma-7b-v1"
)

snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.1", 
    revision="main",
    local_dir=".cache/mistralai/Mistral-7B-v0.1"
)
