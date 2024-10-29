#!/bin/bash

source work/bin/activate
source import-env.sh .env
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
SEED=(0)
MERGE_METHOD=task_arithmetic

gr1=(0.5)
gr2=(0.5)
gr3=(0.5)

python3 merge_llms_instruct_math_code.py \
    --seed $SEED \
    --merge_instruct --merge_math --merge_code \
    --merging_method_name $MERGE_METHOD \
    --scaling_coefficient 1.0 \
    --tensor_parallel_size 1 \
    --weight_mask_rate 0.0 \
    --gradation1 $gr1 \
    --gradation2 $gr2 \
    --gradation3 $gr3 \