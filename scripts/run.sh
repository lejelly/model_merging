#bin/bash


#STRATEGY=( "metagpt_optimize" "metagpt_random" "metagpt_average" "metagpt_all_one" "metagpt_random_normalize" "optuna")
#SEED=( 0 1234 64351 )
#DATASET=( "gsm8k" "mbpp" "ja_mgsm" )
STRATEGY=( "metagpt_average" )
SEED=( 0 )
DATASET=( "gsm8k" )
INTERVAL=10
SAMPLE_RATIO=( 1 3 5 10 15 20 30 40 50 60 70 80 90 100 )

for sample_ratio in ${SAMPLE_RATIO[@]}; do
    for dataset in ${DATASET[@]}; do
        for seed in ${SEED[@]}; do
            for strategy in ${STRATEGY[@]}; do

                while true; do
                    # JOB_ID の行数をカウントしてジョブ数を取得
                    job_count=$(qstat | grep -c "^[0-9]")
                
                    echo "現在のジョブ数: $job_count"
                    
                    if [ "$job_count" -lt 8 ]; then
                        echo "ジョブ数が8未満です。新しいジョブを送信します。"
                        break
                    else
                        echo "ジョブ数が8以上のため、新しいジョブは待機します。"
                        sleep $INTERVAL
                    fi
                done

                qsub -v dataset=$dataset,seed=$seed,strategy=$strategy,sample_ratio=$sample_ratio scripts/meta_gpt.sh
                
            done
        done
    done
done

#bash scripts/meta_gpt.sh gsm8k 0 metagpt_all_one
#qsub -v dataset=gsm8k,seed=0,strategy=metagpt_average,sample_ratio=10 scripts/meta_gpt.sh
