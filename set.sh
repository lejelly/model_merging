#!/bin/bash

cd /home/ubuntu/model_merging
python -m venv work
source work/bin/activate
pip install -r requirements.txt
git config --global user.email jeong@weblab.t.u-tokyo.ac.jp
git config --global user.name lejelly
git config --global credential.helper store
source import-env.sh .lambda_env
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
bash scripts/download.sh

sleep 180
python convert_llama_weights_to_hf.py --input_dir /home/ubuntu/model_merging/.cache/meta-llama/Llama-2-7b --model_size 7B --output_dir /home/ubuntu/model_merging/.cache/meta-llama/Llama-2-7b

wandb login
