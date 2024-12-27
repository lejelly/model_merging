import torch

if torch.cuda.is_available():
    cache_dir = '.cache/huggingface/hub'
else:
    cache_dir = '.cache/huggingface/hub'

print(cache_dir)