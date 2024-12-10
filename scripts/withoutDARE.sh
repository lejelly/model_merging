#!/bin/bash
#PJM -L rscgrp=share-short
#PJM -L gpu=1
#PJM -L elapse=2:00:00
#PJM -j

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

DATASET=$dataset

#MODEL_NAME=WizardLMTeam/WizardMath-7B-V1.1
#MODEL_NAME=augmxnt/shisa-gamma-7b-v1
#MODEL_NAME=GAIR/Abel-7B-002
#MODEL_NAME=tokyotech-llm/Swallow-MS-7b-v0.1
#MODEL_NAME=upaya07/Arithmo2-Mistral-7B

MODEL_NAME=$model_name

#MERGE_METHOD=average_merging
#MERGE_METHOD=task_arithmetic
#MERGE_METHOD=ties_merging

#COMP_FILE_PATH=./results_logging/single_model_inference/${DATASET}/without_DARE.txt
#LOG_RESP_PATH=./results_logging/single_model_inference/${DATASET}/${MODEL_NAME}/withoutDARE_response.json

SEED=(0)
COMP_FILE_PATH=./results_metagpt/single_model_inference/${DATASET}/${MODEL_NAME}.txt
LOG_RESP_PATH=./results_metagpt/single_model_inference/${DATASET}/${MODEL_NAME}.json

python3 inference_llms_instruct_math_code.py \
    --seed $SEED \
    --dataset_name $DATASET \
    --finetuned_model_name $MODEL_NAME \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH 


#pjsub -g gb20 -x dataset="gsm8k",model_name="augmxnt/shisa-gamma-7b-v1" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="human_eval",model_name="augmxnt/shisa-gamma-7b-v1" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="ja_mgsm",model_name="TIGER-Lab/MAmmoTH2-7B" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="ja_mgsm",model_name="Nondzu/Mistral-7B-codealpaca-lora" scripts/withoutDARE.sh

<< COMMENTOUT
MODEL1=WizardMath-7B-V1.1
MODEL2=Arithmo2-Mistral-7B
SEED=(0)
GRADATIONTATE=(1.0)
MERGE_METHOD=task_arithmetic
COMP_FILE_PATH=./results_logging/weighted_task_arithmetic/${DATASET}/${MODEL1}_${MODEL2}/seed${SEED}_gr${GRADATIONTATE}.txt
LOG_RESP_PATH=./results_logging/weighted_task_arithmetic/${DATASET}/${MODEL1}_${MODEL2}/seed${SEED}_gr${GRADATIONTATE}.json

python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --dataset_name $DATASET \
    --merge_math1 --merge_jp1 \
    --merging_method_name $MERGE_METHOD \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH \
    --exclusive_gradation $GRADATIONTATE
COMMENTOUT
