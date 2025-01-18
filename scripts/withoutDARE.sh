#!/bin/bash
#PBS -q short-g
#PBS -l select=1 
#PBS -l walltime=00:30:00
#PBS -W group_list=qj26 
#PBS -j oe

# module load
source import-env.sh .env
module load gcc/12.4.0
module load python/3.10.16 
module load cuda/12.4
module load cudnn/9.5.1.17

# environment setup
cd $PATH_TO_WORKING_DIR
source work/bin/activate
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential

#DATASET=$dataset
DATASET=gsm8k

#MODEL_NAME=mistralai/Mistral-7B-v0.1
#MODEL_NAME=WizardLMTeam/WizardMath-7B-V1.1
#MODEL_NAME=augmxnt/shisa-gamma-7b-v1
#MODEL_NAME=GAIR/Abel-7B-002
#MODEL_NAME=tokyotech-llm/Swallow-MS-7b-v0.1
#MODEL_NAME=upaya07/Arithmo2-Mistral-7B
#MODEL_NAME=TIGER-Lab/MAmmoTH2-7B
#MODEL_NAME=Nondzu/Mistral-7B-codealpaca-lora

#MODEL_NAME=TIGER-Lab/MAmmoTH-7B
#MODEL_NAME=mrm8488/llama-2-coder-7b
#MODEL_NAME=elyza/ELYZA-japanese-Llama-2-7b
#MODEL_NAME=meta-llama/Llama-2-7b

#MODEL_NAME=$model_name
MODEL_NAME=meta-llama/Llama-2-7b

#MERGE_METHOD=average_merging
#MERGE_METHOD=task_arithmetic
#MERGE_METHOD=ties_merging

#COMP_FILE_PATH=./results_logging/single_model_inference/${DATASET}/without_DARE.txt
#LOG_RESP_PATH=./results_logging/single_model_inference/${DATASET}/${MODEL_NAME}/withoutDARE_response.json

SEED=(0)
COMP_FILE_PATH=./results/single_model_inference/base_model/${DATASET}/${MODEL_NAME}.txt
LOG_RESP_PATH=./results/single_model_inference/base_model/${DATASET}/${MODEL_NAME}.json

python3 inference_llms_instruct_math_code.py \
    --seed $SEED \
    --dataset_name $DATASET \
    --finetuned_model_name $MODEL_NAME \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --comp_file_path $COMP_FILE_PATH \
    --log_resp_path $LOG_RESP_PATH \
    --prompt_type "zeroshotcot"


#model_name=("meta-llama/Llama-2-7b")
#dataset=("gsm8k" "mbpp" "ja_mgsm")
#qsub scripts/withoutDARE.sh　-V dataset=gsm8k,model_name="meta-llama/Llama-2-7b" 


#pjsub -g gb20 -x dataset="gsm8k" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="mbpp" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="ja_mgsm" scripts/withoutDARE.sh

#pjsub -g gb20 -x dataset="gsm8k",model_name="WizardLMTeam/WizardMath-7B-V1.1" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="gsm8k",model_name="augmxnt/shisa-gamma-7b-v1" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="gsm8k",model_name="TIGER-Lab/MAmmoTH2-7B" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="gsm8k",model_name="Nondzu/Mistral-7B-codealpaca-lora" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="human_eval",model_name="augmxnt/shisa-gamma-7b-v1" scripts/withoutDARE.sh

#pjsub -g gb20 -x dataset="ja_mgsm",model_name="WizardLMTeam/WizardMath-7B-V1.1" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="ja_mgsm",model_name="TIGER-Lab/MAmmoTH2-7B" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="ja_mgsm",model_name="Nondzu/Mistral-7B-codealpaca-lora" scripts/withoutDARE.sh

#pjsub -g gb20 -x dataset="mbpp",model_name="WizardLMTeam/WizardMath-7B-V1.1" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="mbpp",model_name="TIGER-Lab/MAmmoTH2-7B" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="mbpp",model_name="Nondzu/Mistral-7B-codealpaca-lora" scripts/withoutDARE.sh
#pjsub -g gb20 -x dataset="mbpp",model_name="augmxnt/shisa-gamma-7b-v1" scripts/withoutDARE.sh


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
