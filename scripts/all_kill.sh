#!/bin/bash

# 除外するジョブIDのリスト
exclude_ids=("5362653" "5362654")  # 除外したいジョブIDをここに追加

# pjstatの出力から実行中(RUNNING)と待機中(QUEUED)のジョブIDを取得
job_ids=$(pjstat | awk '($3 == "RUNNING" || $3 == "QUEUED") {print $1}')

# 各ジョブIDに対してpjdelコマンドを実行
for job_id in $job_ids
do
    # ジョブIDが除外リストに含まれているか確認
    if [[ " ${exclude_ids[@]} " =~ " ${job_id} " ]]; then
        echo "Skipping job: $job_id"
        continue
    fi

    echo "Deleting job: $job_id"
    pjdel $job_id
done

echo "All specified jobs have been deleted."