#!/bin/bash

# module load
source import-env.sh .env

DATASETNAME=$1
SEED=$2
MERGE_METHOD=task_arithmetic
STRATEGY=$3

COMP_FILE_PATH=./results_metagpt/${SEED}/math_code_jp/${STRATEGY}/${DATASETNAME}.txt
LOG_RESP_PATH=./results_metagpt/${SEED}/math_code_jp/${STRATEGY}/${DATASETNAME}/json_logs/${STRATEGY}.json

# environment setup
cd $PATH_TO_WORKING_DIR
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN

#INITIAL_LAMBDA_FILEPATH="/home/ubuntu/model_merging/lambdas/initial_lambdas_metagpt_optimize_MAmmoTH2-7B_Mistral-7B-codealpaca-lora_shisa-gamma-7b-v1.csv"
#OPTIMIZED_LAMBDA_FILEPATH="/work/gb20/b20042/model_merging/lambdas/math_code_jp/metagpt_sum/adam_epochs20_lr0.0001_sample100/optimized_lambdas_metagpt_sum_MAmmoTH2-7B_Mistral-7B-codealpaca-lora_shisa-gamma-7b-v1.csv"
#RUN_NAME=seed${SEED}_${STRATEGY}_adam_epochs20_lr0.0001_sample10


python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --merge_math --merge_code --merge_jp \
    --merging_method_name $MERGE_METHOD \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH \
    --dataset_name $DATASETNAME \
    --metagpt \
    --lambda_strategy $STRATEGY \
    --num_epochs 2 \
    --learning_rate 0.001 \
    --num_train_samples 4 \
    --optimizer_type adam 
#    --initial_lambda_filepath $INITIAL_LAMBDA_FILEPATH 
#    --run_name $RUN_NAME \
#    --optimized_lambda_filepath $OPTIMIZED_LAMBDA_FILEPATH 


#bash scripts/meta_gpt.sh gsm8k 0 metagpt_optimize
