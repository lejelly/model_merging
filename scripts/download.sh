#!/bin/bash

#source work/bin/activate

# nohup python download.py --models "TIGER-Lab/MAmmoTH2-7B" &
# nohup python download.py --models "augmxnt/shisa-gamma-7b-v1" &
# nohup python download.py --models "Nondzu/Mistral-7B-codealpaca-lora" &
# nohup python download.py --models "mistralai/Mistral-7B-v0.1" &

nohup python download.py --models "TIGER-Lab/MAmmoTH-7B" > download_TIGER-Lab_MAmmoTH-7B.log &
nohup python download.py --models "mrm8488/llama-2-coder-7b" > download_mrm8488_llama-2-coder-7b.log &
nohup python download.py --models "elyza/ELYZA-japanese-Llama-2-7b" > download_elyza_ELYZA-japanese-Llama-2-7b.log &
nohup python download.py --models "meta-llama/Llama-2-7b" > download_meta-llama_Llama-2-7b.log &
