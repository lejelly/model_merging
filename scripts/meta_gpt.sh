#!/bin/bash
#PJM -L rscgrp=share-debug
#PJM --name proposed_lamdas
#PJM -L gpu=4
#PJM -L elapse=0:30:00
#PJM -j

# module load
source import-env.sh .env
#module load gcc/8.3.1
#module load python/3.10.13
#module load cuda/12.1
#module load cudnn/8.8.1

#DATASETNAME=$dataset
DATASETNAME="human_eval"
SEED=(0)
MERGE_METHOD=task_arithmetic
#MERGE_METHOD=average_merging

COMP_FILE_PATH=./results_metagpt/math_code_jp/gram_matrix/${DATASETNAME}.txt
LOG_RESP_PATH=./results_metagpt/math_code_jp/gram_matrix/${DATASETNAME}.json

# environment setup
cd $PATH_TO_WORKING_DIR
#source work/bin/activate
export TRANSFORMERS_OFFLINE=0
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_ENDPOINT=https://huggingface.co
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

#STRATEGY=$strategy
#STRATEGY="cosine_similarity"
#STRATEGY="graph_laplacian"
#STRATEGY="hierarchical_clustering"
#STRATEGY="attention_based"

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
    --metagpt 


#pjsub -g gb20 -x dataset="gsm8k" scripts/meta_gpt.sh
#pjsub -g gb20 -x dataset="human_eval" scripts/meta_gpt.sh
#pjsub -g gb20 -x dataset="ja_mgsm" scripts/meta_gpt.sh