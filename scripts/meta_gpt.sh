#!/bin/bash

source import-env.sh .env

DATASETNAME=$1
SEED=$2
MERGE_METHOD=task_arithmetic
STRATEGY=$3
GPU_ID=$4

COMP_FILE_PATH=./results_metagpt/${SEED}/llama2_math_llama2_code_llama2_jp/${STRATEGY}/${DATASETNAME}.txt
LOG_RESP_PATH=./results_metagpt/${SEED}/llama2_math_llama2_code_llama2_jp/${STRATEGY}/${DATASETNAME}/json_logs/${STRATEGY}.json

# environment setup
cd $PATH_TO_WORKING_DIR
#source work/bin/activate
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN

#INITIAL_LAMBDA_FILEPATH="/home/ubuntu/model_merging/lambdas/initial_lambdas_metagpt_optimize_MAmmoTH2-7B_Mistral-7B-codealpaca-lora_shisa-gamma-7b-v1.csv"
#OPTIMIZED_LAMBDA_FILEPATH="/home/ubuntu/model_merging/lambdas/seed0/llama2_math_llama2_code_llama2_jp/metagpt_all_one/adam_epochs10_lr0.001_sample4/initial_lambdas_metagpt_all_one_ELYZA-japanese-Llama-2-7b_MAmmoTH-7B_llama-2-coder-7b.csv"
RUN_NAME=llama2_seed${SEED}_${STRATEGY}_${DATASETNAME}_adam_epochs10_lr0.001_sample4

CUDA_VISIBLE_DEVICES=${GPU_ID} python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --llama2_math --llama2_code --llama2_jp \
    --merging_method_name $MERGE_METHOD \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH \
    --dataset_name $DATASETNAME \
    --metagpt \
    --lambda_strategy $STRATEGY \
    --num_epochs 10 \
    --learning_rate 0.001 \
    --num_train_samples 4 \
    --optimizer_type adam \
    --run_name $RUN_NAME
#    --initial_lambda_filepath $INITIAL_LAMBDA_FILEPATH \
#    --optimized_lambda_filepath $OPTIMIZED_LAMBDA_FILEPATH 


#bash scripts/meta_gpt.sh gsm8k 0 metagpt_optimize