#!/bin/bash

mode=${1:-None}
dataset=human_eval
model1=WizardMath-7B-V1.1
model2=Arithmo2-Mistral-7B

#log_path=/work/gb20/b20042/model_merging/save_gen_codes_results/tmp_exlusive_dare
log_path=/work/gb20/b20042/model_merging/save_gen_codes_results/${dataset}/${mode}/${model1}-${model2}

grad1=0.8
grad2=0.2
pjsub -g gb20 -x grad1=${grad1},grad2=${grad2},dataset=${dataset},model1=${model1},model2=${model2},mode=${mode},log_path=${log_path}  scripts/tmp.sh 