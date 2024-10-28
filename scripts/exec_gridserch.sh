#!/bin/bash
#PJM -L rscgrp=share-short
#PJM --name test
#PJM -L gpu=1
#PJM -L elapse=0:50:00
#PJM -j

# module load
source import-env.sh .env
module load gcc/8.3.1
module load python/3.10.13
module load cuda/12.1
module load cudnn/8.8.1

MODE=$mode
DATASET=$dataset
MODEL1=$model1
MODEL2=$model2
SEED=(0)
MERGE_METHOD=task_arithmetic
GRADATIONTATE1=$grad1
GRADATIONTATE2=$grad2
DROP_RATE=(0.5)

#COMP_FILE_PATH=./results_logging/gabage/exclusive_dare/${MODEL1}-${MODEL2}/seed${SEED}.txt
#LOG_RESP_PATH=./results_logging/gabage/exclusive_dare/${MODEL1}-${MODEL2}/seed${SEED}_gr1_${GRADATIONTATE1}_gr2_${GRADATIONTATE2}.json
LOG_RESP_PATH=${log_path}/seed${SEED}_gr1_${GRADATIONTATE1}_gr2_${GRADATIONTATE2}

echo "##############################################"
echo "MODE: $MODE"
echo "MERGE_METHOD: $MERGE_METHOD"
echo "GRADATIONTATE1: $GRADATIONTATE1"
echo "GRADATIONTATE2: $GRADATIONTATE2"
echo "##############################################"

# environment setup
cd $PATH_TO_WORKING_DIR
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

if [ "$MODE" = "exclusive" ]; then
    python3 merge_llms_instruct_math_code.py \
        --seed $SEED \
        --dataset_name $DATASET \
        --merge_math1 --merge_math3 \
        --merging_method_name mask_merging \
        --mask_apply_method $MERGE_METHOD \
        --use_weight_rescale --weight_mask_rate $DROP_RATE \
        --scaling_coefficient 1.0 \
        --tensor_parallel_size 1 \
        --log_resp_path $LOG_RESP_PATH \
        --exclusive_dropout \
        --gradation1 $GRADATIONTATE1 \
        --gradation2 $GRADATIONTATE2 
        #--comp_file_path $COMP_FILE_PATH \
fi

if [ "$MODE" = "dare" ]; then
    python3 merge_llms_instruct_math_code.py \
        --seed $SEED \
        --dataset_name $DATASET \
        --merge_math1 --merge_math3 \
        --merging_method_name mask_merging \
        --mask_apply_method $MERGE_METHOD \
        --use_weight_rescale --weight_mask_rate $DROP_RATE \
        --scaling_coefficient 1.0 \
        --tensor_parallel_size 1 \
        --log_resp_path $LOG_RESP_PATH \
        --gradation1 $GRADATIONTATE1 \
        --gradation2 $GRADATIONTATE2 
        #--comp_file_path $COMP_FILE_PATH \
fi

if [ "$MODE" = "task_arithmetic" ]; then
    python3 merge_llms_instruct_math_code.py \
        --seed $SEED \
        --dataset_name $DATASET \
        --merge_math1 --merge_math3 \
        --merging_method_name $MERGE_METHOD \
        --scaling_coefficient 1.0 \
        --tensor_parallel_size 1 \
        --weight_mask_rate 0.0 \
        --log_resp_path $LOG_RESP_PATH \
        --gradation1 $GRADATIONTATE1 \
        --gradation2 $GRADATIONTATE2 
        #--comp_file_path $COMP_FILE_PATH \
fi
