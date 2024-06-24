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

DATASET=ja_mgsm
DROP_METHOD=dare

MODELS=(
    "WizardLMTeam/WizardMath-7B-V1.1"
    "augmxnt/shisa-gamma-7b-v1"
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

echo "All inferences completed"

#python3 merge_llms_instruct_math_code.py \
#    --merge_jp1 --merge_math1 \
#    --merging_method_name mask_merging \
#    --mask_apply_method task_arithmetic \
#    --use_weight_rescale --weight_mask_rate 0.05 \
#    --scaling_coefficient 1.0 \
#    --tensor_parallel_size 1 \
#    --comp_file_path $COMP_FILE_PATH

#python3 merge_llms_instruct_math_code.py \
#    --merge_jp1 --merge_math1 \
#    --merging_method_name mask_merging \
#    --mask_apply_method ties_merging \
#    --use_weight_rescale --weight_mask_rate 0.05 \
#    --scaling_coefficient 1.0 \
#    --tensor_parallel_size 1 \
#    --comp_file_path $COMP_FILE_PATH

