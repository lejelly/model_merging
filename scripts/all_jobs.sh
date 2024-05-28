#!/bin/bash

MODEL_NAME_1=WizardLMTeam/WizardMath-7B-V1.1
MODEL_NAME_2=augmxnt/shisa-gamma-7b-v1
MODEL_NAME_3=GAIR/Abel-7B-002
MODEL_NAME_4=tokyotech-llm/Swallow-MS-7b-v0.1

MODEL_SETS=($MODEL_NAME_1 $MODEL_NAME_2 $MODEL_NAME_3 $MODEL_NAME_4)

## w/o DARE ##########################################################
SCRIPT_TYPE=withoutDARE
for MODEL in ${MODEL_SETS[@]}
do
    PROMPT=zeroshotcot
    qsub -g gcb50389 -N log_cot_${SCRIPT_TYPE}_${PROMPT} scripts/${SCRIPT_TYPE}.sh $MODEL $PROMPT 
    PROMPT=fewshotcot
    qsub -g gcb50389 -N log_cot_${SCRIPT_TYPE}_${PROMPT} scripts/${SCRIPT_TYPE}.sh $MODEL $PROMPT
done

######################################################################
MODEL_NAME=WizardLMTeam/WizardMath-7B-V1.1
PROMPT=zeroshotcot
DROP_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)
for DROP_RATE in ${DROP_RATES[@]}
do
    ## Drop Only #####################################################
    SCRIPT_TYPE=dropOnly
    qsub -g gcb50389 -N log_${SCRIPT_TYPE}_${DROP_RATE}_wizardMath scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT $DROP_RATE  
    ## Magnitude-Based Pruning #######################################
    SCRIPT_TYPE=magnitude
    qsub -g gcb50389 -N log_${SCRIPT_TYPE}_${DROP_RATE}_wizardMath scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT $DROP_RATE  
    ## Masking Fine-Tuned Parameters #################################
    SCRIPT_TYPE=finetuned
    qsub -g gcb50389 -N log_${SCRIPT_TYPE}_${DROP_RATE}_wizardMath scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT $DROP_RATE  
    ## DARE ##########################################################
    SCRIPT_TYPE=dare
    qsub -g gcb50389 -N log_${SCRIPT_TYPE}_${DROP_RATE}_wizardMath scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT $DROP_RATE  
done