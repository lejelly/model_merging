#!/bin/bash
#PJM -L rscgrp=debug-a
#PJM -L node=1
#PJM -L elapse=0:20:00
#PJM -j

#DROP_RATE=$PJM_JOBENV_DROP_RATE

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

DATASET=ja_mgsm
DROP_METHOD=dare

<< COMMENTOUT
DROP_RATE=(0.5)
SEED=(0)
MODEL_NAME=GAIR/Abel-7B-002
COMP_FILE_PATH=./results_logging/single_model_inference/${DATASET}/${MODEL_NAME}/${DROP_METHOD}/.txt

python3 inference_llms_instruct_math_code.py \
    --seed $SEED \
    --dataset_name $DATASET \
    --finetuned_model_name $MODEL_NAME \
    --tensor_parallel_size 1 \
    --use_weight_rescale --weight_mask_rate $DROP_RATE \
    --drop_method $DROP_METHOD \
    --comp_file_path $COMP_FILE_PATH 


#MODEL_NAME=GAIR/Abel-7B-002
COMP_FILE_PATH=./results/single_model_inference/${DATASET}/${DROP_METHOD}/${MODEL_NAME}.txt

python3 inference_llms_instruct_math_code.py \
    --dataset_name $DATASET \
    --finetuned_model_name $MODEL_NAME \
    --tensor_parallel_size 1 \
    --weight_mask_rate $DROP_RATE \
    --use_weight_rescale \
    --drop_method $DROP_METHOD \
    --comp_file_path $COMP_FILE_PATH \ 
    --exclusive_dropout \ 
    --subordinate_mask


COMP_FILE_PATH=./results_logging/single_model_inference/${DATASET}/dare.txt
LOG_RESP_PATH=./results_logging/single_model_inference/${DATASET}/${DROP_METHOD}/dare_response.json
MODELS=(
    "GAIR/Abel-7B-002"
)


DROP_RATES=($(seq 0.0 0.1 1.0) 0.99)
for MODEL_NAME in "${MODELS[@]}"; do
    echo "Starting inference for model: $MODEL_NAME"
    COMP_FILE_PATH=./results/single_model_inference/${DATASET}/${DROP_METHOD}/${MODEL_NAME}.txt

    for DROP_RATE in "${DROP_RATES[@]}"; do
        echo "Running with drop rate: $DROP_RATE"
        python3 inference_llms_instruct_math_code.py \
            --dataset_name $DATASET \
            --finetuned_model_name $MODEL_NAME \
            --tensor_parallel_size 1 \
            --weight_mask_rate $DROP_RATE \
            --use_weight_rescale \
            --drop_method $DROP_METHOD \
            --comp_file_path $COMP_FILE_PATH 
    done
done
COMMENTOUT

MODEL1=WizardMath-7B-V1.1
MODEL2=shisa-gamma-7b-v1

MERGE_METHOD=task_arithmetic
#MERGE_METHOD=ties_merging
DROP_RATE=(0.5)
SEED=(0)
GRADATIONTATE=(0.2)
COMP_FILE_PATH=./results_logging/merged_model_inference/${DATASET}/weighted_task_arithmetic/seed${SEED}_gr${GRADATIONTATE}.txt
LOG_RESP_PATH=./results_logging/merged_model_inference/${DATASET}/weighted_task_arithmetic/${MODEL1}_${MODEL2}/${MERGE_METHOD}/seed${SEED}_gr${GRADATIONTATE}.json
#COMP_FILE_PATH=./results_logging/exclusive_dare/${DATASET}/${MODEL1}_${MODEL2}/exclusive_dare_seed${SEED}_gr${GRADATIONTATE}.txt
#LOG_RESP_PATH=./results_logging/exclusive_dare/${DATASET}/${MODEL1}_${MODEL2}/dare_response_seed${SEED}_gr${GRADATIONTATE}.json

python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --merge_math1 --merge_jp1 \
    --merging_method_name mask_merging \
    --mask_apply_method $MERGE_METHOD \
    --use_weight_rescale --weight_mask_rate $DROP_RATE \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH \
    --exclusive_gradation $GRADATIONTATE

<< COMMENTOUT
#EXLUSIVE SETTING
python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --merge_math1 --merge_jp1 \
    --merging_method_name mask_merging \
    --mask_apply_method $MERGE_METHOD \
    --use_weight_rescale --weight_mask_rate $DROP_RATE \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH \
    --exclusive_dropout \
    --exclusive_gradation $GRADATIONTATE
COMMENTOUT