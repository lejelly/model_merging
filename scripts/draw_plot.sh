#!/bin/bash

#[ WizardLMTeam/WizardMath-7B-V1.1 augmxnt/shisa-gamma-7b-v1 GAIR/Abel-7B-002 tokyotech-llm/Swallow-MS-7b-v0.1 ]
MODEL_NAME=augmxnt/shisa-gamma-7b-v1
#METHODS=(dare droponly finetuned magnitude)
METHODS=(dare)

for METHOD in ${METHODS[@]}
do
    ## DARE ##########################################################
    python draw_figures/comp_single_model.py --model_name $MODEL_NAME --method_name $METHOD
done


