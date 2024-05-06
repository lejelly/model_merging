#!/bin/bash
#$ -l rt_AG.small=1
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

MODEL_NAME=$1
PROMPT_TYPE=$2
DROP_RATE=$3
DATASET=gsm8k
COMP_FILE_PATH=./results/single_model_inference/${PROMPT_TYPE}/gsm8k_finetuned_${MODEL_NAME}.txt

# module load
source /etc/profile.d/modules.sh
module load python/3.11/3.11.9 
module load cuda/12.1/12.1.1 
module load cudnn/8.9/8.9.7 

# start inference
cd /home/acf15429bz/model_merging/model_merging
source work/bin/activate
HUGGINGFACE_TOKEN=
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
python inference_llms_instruct_math_code.py \
    --dataset_name $DATASET \
    --finetuned_model_name $MODEL_NAME \
    --tensor_parallel_size 1 \
    --weight_mask_rate $DROP_RATE \
    --use_weight_rescale \
    --weight_format finetuned_weight \
    --comp_file_path $COMP_FILE_PATH \
    --prompt_type $PROMPT_TYPE
