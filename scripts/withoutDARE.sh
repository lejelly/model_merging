#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -j

#MODEL_NAME=WizardLMTeam/WizardMath-7B-V1.1
#MODEL_NAME=augmxnt/shisa-gamma-7b-v1
#MODEL_NAME=GAIR/Abel-7B-002
#MODEL_NAME=tokyotech-llm/Swallow-MS-7b-v0.1
#MODEL_NAME=upaya07/Arithmo2-Mistral-7B

DATASET=ja_mgsm
COMP_FILE_PATH=./results/merged_model_inference/${DATASET}/without_DARE.txt

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
#python3 inference_llms_instruct_math_code.py \
#    --dataset_name $DATASET \
#    --finetuned_model_name $MODEL_NAME \
#    --tensor_parallel_size 1 \
#    --weight_mask_rate 0.0 \
#    --comp_file_path $COMP_FILE_PATH 
#    --prompt_type $PROMPT_TYPE

python3 merge_llms_instruct_math_code.py \
    --merge_jp1 --merge_math1 \
    --merging_method_name task_arithmetic \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH 

python3 merge_llms_instruct_math_code.py \
    --merge_jp1 --merge_math1 \
    --merging_method_name ties_merging \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH 