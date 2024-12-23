#!/bin/bash
#PJM -L rscgrp=share-short
#PJM --name gram_matrix_4_models
#PJM -L gpu=2
#PJM -L elapse=1:00:00
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

COMP_FILE_PATH=./results_metagpt/math_code_jp/Metagpt_strict/${DATASETNAME}.txt
LOG_RESP_PATH=./results_metagpt/math_code_jp/Metagpt_strict/${DATASETNAME}/json_logs/Metagpt_strict.json

# environment setup
cd $PATH_TO_WORKING_DIR
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

STRATEGY=$strategy
#STRATEGY="cosine_similarity"
#STRATEGY="graph_laplacian"
#STRATEGY="hierarchical_clustering"
#STRATEGY="attention_based"

python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --merge_math1 --merge_math2 --merge_code --merge_jp \
    --merging_method_name $MERGE_METHOD \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH \
    --dataset_name $DATASETNAME \
    --metagpt \
    --lambda_strategy $STRATEGY


#pjsub -g gb20 -x dataset="gsm8k",strategy="metagpt" scripts/meta_gpt.sh
#pjsub -g gb20 -x dataset="human_eval",strategy="metagpt" scripts/meta_gpt.sh
#pjsub -g gb20 -x dataset="ja_mgsm",strategy="metagpt" scripts/meta_gpt.sh

#pjsub -g gb20 -x dataset="gsm8k",strategy="metagpt_strict" scripts/meta_gpt.sh
#pjsub -g gb20 -x dataset="human_eval",strategy="metagpt_strict" scripts/meta_gpt.sh
#pjsub -g gb20 -x dataset="ja_mgsm",strategy="metagpt_strict" scripts/meta_gpt.sh
