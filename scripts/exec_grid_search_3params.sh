#!/bin/bash
#PJM -L rscgrp=share-short
#PJM --name log_metagpt
#PJM -L gpu=2
#PJM -L elapse=0:15:00
#PJM -j

# module load
source import-env.sh .env
module load gcc/8.3.1
module load python/3.10.13
module load cuda/12.1
module load cudnn/8.8.1

SEED=(0)
MERGE_METHOD=task_arithmetic
GRADATIONTATE1=$grad1
GRADATIONTATE2=$grad2
GRADATIONTATE3=$grad3

COMP_FILE_PATH=./results_logging/metagpt/seed_${SEED}_gr1_${GRADATIONTATE1}_gr2_${GRADATIONTATE2}_gr3_${GRADATIONTATE3}.txt
LOG_RESP_PATH=./results_logging/metagpt/seed${SEED}_gr1_${GRADATIONTATE1}_gr2_${GRADATIONTATE2}_gr3_${GRADATIONTATE3}.json
#LOG_RESP_PATH=${log_path}/seed${SEED}_gr1_${GRADATIONTATE1}_gr2_${GRADATIONTATE2}

echo "##############################################"
echo "MERGE_METHOD: $MERGE_METHOD"
echo "GRADATIONTATE1: $GRADATIONTATE1"
echo "GRADATIONTATE2: $GRADATIONTATE2"
echo "GRADATIONTATE2: $GRADATIONTATE3"
echo "##############################################"

# environment setup
cd $PATH_TO_WORKING_DIR
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

<< COMMENTOUT
python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --merge_instruct --merge_code \
    --merging_method_name $MERGE_METHOD \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --gradation1 $GRADATIONTATE1 \
    --gradation2 $GRADATIONTATE2 \
    --gradation3 $GRADATIONTATE3 \
    --comp_file_path $COMP_FILE_PATH \
    --dataset_name $DATASETNAME
COMMENTOUT

python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --merge_instruct --merge_math --merge_code \
    --merging_method_name $MERGE_METHOD \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --gradation1 $GRADATIONTATE1 \
    --gradation2 $GRADATIONTATE2 \
    --gradation3 $GRADATIONTATE3 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH 

