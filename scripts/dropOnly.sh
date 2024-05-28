#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=0:20:00
#$ -j y
#$ -cwd

MODEL_NAME=$1
PROMPT_TYPE=$2
DROP_RATE=$3
DATASET=gsm8k
COMP_FILE_PATH=./results/single_model_inference/${PROMPT_TYPE}/gsm8k_droponly_${MODEL_NAME}.txt

# module load
source import-env.sh .env
source /etc/profile.d/modules.sh
module load python/3.10/3.10.14
module load cuda/12.1/12.1.1 
module load cudnn/8.9/8.9.7 

# environment setup
cd $PATH_TO_WORKING_DIR
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
export CUDA_HOME=$CUDA_HOME_PATH

# start inference
python inference_llms_instruct_math_code.py \
    --dataset_name $DATASET \
    --finetuned_model_name $MODEL_NAME \
    --tensor_parallel_size 1 \
    --weight_mask_rate $DROP_RATE \
    --comp_file_path $COMP_FILE_PATH \
    --prompt_type $PROMPT_TYPE