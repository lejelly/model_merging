#bin/bash


STRATEGY=( "metagpt_optimize" "metagpt_random" "metagpt_average" "metagpt_all_one" "metagpt_random_normalize" "optuna")
#SEED=( 0 1234 64351 )
#DATASET=( "gsm8k" "mbpp" "ja_mgsm" )
SEED=( 0 )
DATASET=( "gsm8k" )
INTERVAL=10

for dataset in ${DATASET[@]}; do
    for seed in ${SEED[@]}; do
        for strategy in ${STRATEGY[@]}; do

            bash scripts/meta_gpt.sh $dataset $seed $strategy 
        
        done
    done
done

#bash scripts/meta_gpt.sh gsm8k 0 metagpt_all_one
#qsub -v dataset=gsm8k,seed=0,strategy=metagpt_optimize scripts/meta_gpt.sh