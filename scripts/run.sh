#bin/bash

STRATEGY=( "optuna" )
SEED=( 0 1234 64351 )

for strategy in ${STRATEGY[@]}; do
    for seed in ${SEED[@]}; do

        bash scripts/meta_gpt.sh gsm8k $seed $strategy
            
    done
done

#bash scripts/meta_gpt.sh gsm8k 0 metagpt_all_one