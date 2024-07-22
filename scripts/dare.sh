#!/bin/bash
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -j

DATASET=ja_mgsm
DROP_METHOD=dare
MODEL1=WizardMath-7B-V1.1
MODEL2=shisa-gamma-7b-v1

#MERGE_METHOD=task_arithmetic
#MERGE_METHOD=ties_merging

COMP_FILE_PATH=./results_logging/merged_model_inference/${DATASET}/dare.txt
LOG_RESP_PATH=./results_logging/merged_model_inference/${DATASET}/${MODEL1}_${MODEL2}/${MERGE_METHOD}/dare_response.json


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

#MODELS=(
#    "WizardLMTeam/WizardMath-7B-V1.1"
#    "augmxnt/shisa-gamma-7b-v1"
#)

#DROP_RATES=($(seq 0.0 0.1 1.0) 0.99)
#for MODEL_NAME in "${MODELS[@]}"; do
#    echo "Starting inference for model: $MODEL_NAME"
#    COMP_FILE_PATH=./results/single_model_inference/${DATASET}/${DROP_METHOD}/${MODEL_NAME}.txt
#
#    for DROP_RATE in "${DROP_RATES[@]}"; do
#        echo "Running with drop rate: $DROP_RATE"
#        python3 inference_llms_instruct_math_code.py \
#            --dataset_name $DATASET \
#            --finetuned_model_name $MODEL_NAME \
#            --tensor_parallel_size 1 \
#            --weight_mask_rate $DROP_RATE \
#            --use_weight_rescale \
#            --drop_method $DROP_METHOD \
#            --comp_file_path $COMP_FILE_PATH
#    done
#done

#echo "All inferences completed"

DROP_RATE=(0.1)

python3 merge_llms_instruct_math_code.py \
    --merge_jp1 --merge_math1 \
    --merging_method_name mask_merging \
    --mask_apply_method $MERGE_METHOD \
    --use_weight_rescale --weight_mask_rate $DROP_RATE \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --comp_file_path $COMP_FILE_PATH\
    --log_resp_path $LOG_RESP_PATH

