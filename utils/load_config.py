import torch

if torch.cuda.is_available():
    cache_dir = '/scratch/acf15429bz/model_merge'
else:
    cache_dir = '/scratch/acf15429bz/model_merge'
