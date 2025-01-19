import torch

if torch.cuda.is_available():
    cache_dir = '/home/ubuntu/model_merging/.cache'
else:
    cache_dir = '/home/ubuntu/model_merging/.cache'

print(cache_dir)