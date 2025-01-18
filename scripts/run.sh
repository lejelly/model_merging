#bin/bash


STRATEGY=( "metagpt_optimize" "metagpt_random" "metagpt_average" "metagpt_all_one" "metagpt_random_normalize" )
#STRATEGY=( "optuna" )
SEED=( 0 1234 64351 )
INTERVAL=10

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

        qsub -v dataset=gsm8k,seed=$seed,strategy=$strategy scripts/meta_gpt.sh
            
    done
done

#bash scripts/meta_gpt.sh gsm8k 0 metagpt_all_one