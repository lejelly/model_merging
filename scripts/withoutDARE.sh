#!/bin/bash
#PJM -L rscgrp=short-a  
#PJM -L node=1
#PJM -L elapse=0:20:00
#PJM -j

MODEL_NAME=upaya07/Arithmo2-Mistral-7B
DATASET=ja_mgsm
COMP_FILE_PATH=./results/single_model_inference/${DATASET}/without_DARE.txt

# module load
source import-env.sh .env
module load gcc/8.3.1
module load python/3.10.13
module load cuda/12.1
module load cudnn/8.8.1

# environment setup
cd $PATH_TO_WORKING_DIR
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

# start inference
python3 inference_llms_instruct_math_code.py \
    --dataset_name $DATASET \
    --finetuned_model_name $MODEL_NAME \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH 
#    --prompt_type $PROMPT_TYPE