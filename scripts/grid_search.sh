#!/bin/bash

# パラメータの範囲
<< COMMENTOUT
grad1=(0.0)
for grad2 in $(seq 0.0 0.2 1.0); do
  # 実行コマンド
  pjsub -g gb20 -x grad1=${grad1},grad2=${grad2}  scripts/tmp.sh 
done


sleep 600
grad1=(0.2)
for grad2 in $(seq 0.0 0.2 1.0); do
  # 実行コマンド
  pjsub -g gb20 -x grad1=${grad1},grad2=${grad2}  scripts/tmp.sh 
done
COMMENTOUT

sleep 100
grad1=(0.4)
for grad2 in $(seq 0.0 0.2 1.0); do
  # 実行コマンド
  pjsub -g gb20 -x grad1=${grad1},grad2=${grad2}  scripts/tmp2.sh 
done

sleep 1200
grad1=(0.6)
for grad2 in $(seq 0.0 0.2 1.0); do
  # 実行コマンド
  pjsub -g gb20 -x grad1=${grad1},grad2=${grad2}  scripts/tmp2.sh 
done

<< COMMENTOUT
sleep 1300
grad1=(0.8)
for grad2 in $(seq 0.0 0.2 1.0); do
  # 実行コマンド
  pjsub -g gb20 -x grad1=${grad1},grad2=${grad2}  scripts/tmp.sh 
done


# DONE: 0.0 0.2 0.4 0.6 0.8
grad1=(1.0)
for grad2 in $(seq 0.0 0.2 1.0); do
  # 実行コマンド
  pjsub -g gb20 -x grad1=${grad1},grad2=${grad2}  scripts/tmp.sh 
done
COMMENTOUT