#!/bin/bash
#PJM -L rscgrp=share
#PJM --name metagpt_average_lambda
#PJM -L gpu=2
#PJM -L elapse=5:00:00
#PJM -j

# module load
source import-env.sh .env
module load gcc/8.3.1
module load python/3.10.13
module load cuda/12.1
module load cudnn/8.8.1

DATASETNAME=$dataset
SEED=(0)
MERGE_METHOD=task_arithmetic
#MERGE_METHOD=average_merging

COMP_FILE_PATH=./results_metagpt/math_code_jp/metagpt_average_lambda/${DATASETNAME}.txt
LOG_RESP_PATH=./results_metagpt/math_code_jp/metagpt_average_lambda/${DATASETNAME}/json_logs/Metagpt_blackbox.json

# environment setup
cd $PATH_TO_WORKING_DIR
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

STRATEGY=$strategy
#STRATEGY="cosine_similarity"
#STRATEGY="graph_laplacian"
#STRATEGY="hierarchical_clustering"
#STRATEGY="attention_based"

INITIAL_LAMBDA_FILEPATH="/work/gb20/b20042/model_merging/lambdas/math_code_jp/metagpt_average_lambda/initial_lambdas_metagpt_alpha_MAmmoTH2-7B_Mistral-7B-codealpaca-lora_shisa-gamma-7b-v1.csv"
#OPTIMIZED_LAMBDA_FILEPATH="/work/gb20/b20042/model_merging/lambdas/math_code_jp/metagpt_optimize/sgd_epochs4_lr0.001_sample100/optimized_lambdas_metagpt_optimize_MAmmoTH2-7B_Mistral-7B-codealpaca-lora_shisa-gamma-7b-v1.csv"
#RUN_NAME="sgd_epochs4_lr0.001_sample100"

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
    --num_epochs 10 \
    --learning_rate 0.0001 \
    --num_train_samples 100 \
    --optimizer_type adam
#    --run_name $RUN_NAME \
#    --optimized_lambda_filepath $OPTIMIZED_LAMBDA_FILEPATH 


#pjsub -g gb20 -x dataset="gsm8k",strategy="metagpt_average_lambda" scripts/meta_gpt_average_lambda.sh
#pjsub -g gb20 -x dataset="mbpp",strategy="metagpt_average_lambda" scripts/meta_gpt_average_lambda.sh
#pjsub -g gb20 -x dataset="ja_mgsm",strategy="metagpt_average_lambda" scripts/meta_gpt_average_lambda.sh
