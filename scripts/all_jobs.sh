#!/bin/bash

## w/o DARE #####################################################
SCRIPT_TYPE=withoutDARE
PROMPT=zeroshotcot

qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_${PROMPT} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.wizardlm_${PROMPT} scripts/${SCRIPT_TYPE}.sh WizardLMTeam/WizardLM-13B-V1.0 $PROMPT
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.code-alpaca_${PROMPT} scripts/${SCRIPT_TYPE}.sh layoric/llama-2-13b-code-alpaca $PROMPT

PROMPT=fewshotcot
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_${PROMPT}_${SCRIPT_TYPE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.wizardlm_${PROMPT}_${SCRIPT_TYPE} scripts/${SCRIPT_TYPE}.sh WizardLMTeam/WizardLM-13B-V1.0 $PROMPT
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.code-alpaca_${PROMPT}_${SCRIPT_TYPE} scripts/${SCRIPT_TYPE}.sh layoric/llama-2-13b-code-alpaca $PROMPT

## Drop Only #####################################################
SCRIPT_TYPE=dropOnly
PROMPT=zeroshotcot
DROP_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)

for DROP_RATE in ${DROP_RATES[@]}
do
    qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_dr${DROP_RATE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT $DROP_RATE  
    sleep 60
done

## Magnitude-Based Pruning #####################################################
SCRIPT_TYPE=magnitude
PROMPT=zeroshotcot
DROP_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)

for DROP_RATE in ${DROP_RATES[@]}
do
    qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_dr${DROP_RATE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT $DROP_RATE  
    sleep 60
done

## Masking Fine-Tuned Parameters #####################################################
SCRIPT_TYPE=finetuned
PROMPT=zeroshotcot
DROP_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)

for DROP_RATE in ${DROP_RATES[@]}
do
    qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_dr${DROP_RATE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT $DROP_RATE  
    sleep 60
done

## DARE #####################################################

SCRIPT_TYPE=dare
PROMPT=zeroshotcot
DROP_RATES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99)

for DROP_RATE in ${DROP_RATES[@]}
do
    qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_dr${DROP_RATE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT $DROP_RATE  
    sleep 60
done