#!/bin/bash

# 実行間隔（秒）
INTERVAL=10
DATASETS="gsm8k human_eval ja_mgsm"

<< COMMENTOUT
STRATEGY="multitask_learning"
for dataset in $DATASETS; do
  for strategy in $STRATEGY; do
    pjsub -g gb20 -x dataset=${dataset},strategy=${strategy} scripts/meta_gpt.sh
  done
done
COMMENTOUT

for dataset in $DATASETS; do
  for grad1 in $(seq 0.2 0.2 0.6); do
    for grad2 in $(seq 0.2 0.2 0.6); do
      for grad3 in $(seq 0.2 0.2 0.6); do
        # スキップ条件の確認
        if [ "$dataset" == "gsm8k" ]; then
          continue
        fi
        # スキップ条件の確認
        if { [ "$dataset" == "human_eval" ] && { [ "$grad1" == "0.2" ] || [ "$grad1" == "0.4" ]; }; } || \
           { [ "$dataset" == "human_eval" ] && [ "$grad1" == "0.6" ] && [ "$grad2" == "0.2" ]; }; then
          echo "Skipping job for dataset=$dataset, grad1=$grad1, grad2=$grad2, grad3=$grad3"
          continue
        fi
        # ジョブ数が12未満になるまで待機
        while true; do
          # JOB_ID の行数をカウントしてジョブ数を取得
          job_count=$(pjstat | grep -c "^[0-9]")

          echo "現在のジョブ数: $job_count"

          # ジョブ数が12未満であれば新しいジョブを送信
          if [ "$job_count" -lt 12 ]; then
            echo "ジョブ数が12未満です。新しいジョブを送信します。"
            break
          else
            echo "ジョブ数が12以上のため、新しいジョブは待機します。"
          fi

          # 指定した間隔だけ待機
          sleep $INTERVAL
        done

        # 実行コマンド
        pjsub -g gb20 -x dataset=$dataset,grad1=$grad1,grad2=$grad2,grad3=$grad3 scripts/meta_gpt.sh
        
      done
    done
  done
done


#pjsub -g gb20 -x dataset="human_eval",strategy="graph_laplacian" scripts/meta_gpt.sh
#pjsub -g gb20 -x dataset="human_eval",strategy="hierarchical_clustering" scripts/meta_gpt.sh
#pjsub -g gb20 -x dataset="human_eval",strategy="attention_based" scripts/meta_gpt.sh

#pjsub -g gb20 -x dataset="ja_mgsm",grad1=0.8,grad2=0.8,grad3=0.8 scripts/meta_gpt.sh