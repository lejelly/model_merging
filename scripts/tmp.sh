#!/bin/bash

## w/o DARE #####################################################
SCRIPT_TYPE=withoutDARE
PROMPT=zeroshotcot
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_${PROMPT} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT
## Drop Only #####################################################
SCRIPT_TYPE=dropOnly
PROMPT=zeroshotcot
DROP_RATE=0.1
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_dr${DROP_RATE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT $DROP_RATE  
## Magnitude-Based Pruning #####################################################
SCRIPT_TYPE=magnitude
PROMPT=zeroshotcot
DROP_RATES=0.2
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_dr${DROP_RATE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT $DROP_RATE  
## Masking Fine-Tuned Parameters #####################################################
SCRIPT_TYPE=finetuned
PROMPT=zeroshotcot
DROP_RATES=0.3
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_dr${DROP_RATE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT $DROP_RATE  
## DARE #####################################################
SCRIPT_TYPE=dare
PROMPT=zeroshotcot
DROP_RATES=0.4
qsub -g gcb50389 -N log.${SCRIPT_TYPE}.xwin_dr${DROP_RATE} scripts/${SCRIPT_TYPE}.sh Xwin-LM/Xwin-Math-13B-V1.0 $PROMPT $DROP_RATE  