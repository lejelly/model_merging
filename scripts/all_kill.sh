#!/bin/bash

# pjstatの出力から実行中(RUNNING)と待機中(QUEUED)のジョブIDを取得
job_ids=$(pjstat | awk '($3 == "RUNNING" || $3 == "QUEUED") {print $1}')

# 各ジョブIDに対してpjdelコマンドを実行
for job_id in $job_ids
do
    echo "Deleting job: $job_id"
    pjdel $job_id
done

echo "All specified jobs have been deleted."