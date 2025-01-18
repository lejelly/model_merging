import torch

if torch.cuda.is_available():
    cache_dir = '/work/gj26/b20042/model_merging/.cache'
else:
    cache_dir = '/work/gj26/b20042/model_merging/.cache'

print(cache_dir)