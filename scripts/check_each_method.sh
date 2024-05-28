#!/bin/bash

MODEL_NAME=WizardLMTeam/WizardMath-7B-V1.1
PROMPT=zeroshotcot
DROP_RATE=0.6
## w/o DARE #####################################################
SCRIPT_TYPE=withoutDARE
qsub -g gcb50389 -N log.${SCRIPT_TYPE} scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT
## Drop Only #####################################################
SCRIPT_TYPE=dropOnly
qsub -g gcb50389 -N log.${SCRIPT_TYPE} scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT $DROP_RATE  
## Magnitude-Based Pruning #####################################################
SCRIPT_TYPE=magnitude
qsub -g gcb50389 -N log.${SCRIPT_TYPE} scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT $DROP_RATE  
## Masking Fine-Tuned Parameters #####################################################
SCRIPT_TYPE=finetuned
qsub -g gcb50389 -N log.${SCRIPT_TYPE} scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT $DROP_RATE  
## DARE #####################################################
SCRIPT_TYPE=dare 
qsub -g gcb50389 -N log.${SCRIPT_TYPE} scripts/${SCRIPT_TYPE}.sh $MODEL_NAME $PROMPT $DROP_RATE   
