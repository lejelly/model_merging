import torch

if torch.cuda.is_available():
    cache_dir = '.cache'
else:
    cache_dir = '.cache'

print(cache_dir)