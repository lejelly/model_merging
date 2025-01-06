#!/bin/bash

nohup python download.py --models "TIGER-Lab/MAmmoTH2-7B" &
nohup python download.py --models "augmxnt/shisa-gamma-7b-v1" &
nohup python download.py --models "Nondzu/Mistral-7B-codealpaca-lora" &
nohup python download.py --models "mistralai/Mistral-7B-v0.1" &
