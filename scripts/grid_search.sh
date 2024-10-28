#!/bin/bash

mode=${1:-None}
dataset=human_eval
model1=WizardMath-7B-V1.1
model2=Arithmo2-Mistral-7B
log_path=/work/gb20/b20042/model_merging/save_gen_codes_results/${dataset}/${mode}/${model1}-${model2}

# 実行間隔（秒）
INTERVAL=10

# mode が None の場合はエラーメッセージを表示して終了
if [ "$mode" == "None" ]; then
  echo "エラー: mode を指定してください。"
  echo "使用法: ./script_name.sh <mode>"
  exit 1
fi

for grad1 in $(seq 0.0 0.2 1.0); do
  for grad2 in $(seq 0.0 0.2 1.0); do
    
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
    pjsub -g gb20 -x grad1=${grad1},grad2=${grad2},dataset=${dataset},model1=${model1},model2=${model2},mode=${mode},log_path=${log_path}  scripts/tmp.sh 
    
  done
done
