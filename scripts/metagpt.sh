#!/bin/bash

source work/bin/activate
source import-env.sh .env
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
mode=task_arithmetic
model1=WizardMath-7B-V1.1
MODEL1=$model1
SEED=(0)
MERGE_METHOD=task_arithmetic

dataset=human_eval
DATASET=$dataset

if [ "$dataset" = "human_eval" ]; then
    model2=Arithmo2-Mistral-7B
    MODEL2=$model2
    DATASET=$dataset
    log_path=/p-shared-4/tmp/model_merging/save_gen_codes_results/${dataset}/${mode}/${model1}-${model2}
    LOG_RESP_PATH=${log_path}/seed${SEED}_metagpt
    python3 merge_llms_instruct_math_code.py \
        --seed $SEED \
        --dataset_name $DATASET \
        --merge_math1 --merge_math3 \
        --merging_method_name $MERGE_METHOD \
        --scaling_coefficient 1.0 \
        --tensor_parallel_size 1 \
        --weight_mask_rate 0.0 \
        --log_resp_path $LOG_RESP_PATH \
        --metagpt 
fi

if [ "$dataset" = "ja_mgsm" ]; then
    model2=shisa-gamma-7b-v1
    MODEL2=$model2
    DATASET=$dataset
    COMP_FILE_PATH=./results_logging/task_arithmetic/${MODEL1}-${MODEL2}/seed${SEED}.txt
    LOG_RESP_PATH=./results_logging/task_arithmetic/${MODEL1}-${MODEL2}/seed${SEED}_metagpt.json

    python3 merge_llms_instruct_math_code.py \
        --seed $SEED \
        --dataset_name $DATASET \
        --merge_math1 --merge_jp1 \
        --merging_method_name $MERGE_METHOD \
        --scaling_coefficient 1.0 \
        --tensor_parallel_size 1 \
        --weight_mask_rate 0.0 \
        --log_resp_path $LOG_RESP_PATH \
        --metagpt \
        --comp_file_path $COMP_FILE_PATH
fi