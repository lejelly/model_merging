#!/bin/bash

# 対象ディレクトリのパスを変数に代入
TARGET_DIR="/work/gb20/b20042/model_merging/save_gen_codes_results/human_eval/exclusive/WizardMath-7B-V1.1-Arithmo2-Mistral-7B"

# ファイル数をカウントする変数を初期化
file_count=0

# 対象ディレクトリ内のすべての .jsonl ファイルに対して処理を実行
for file in "$TARGET_DIR"/*.jsonl; do
    # ファイルが存在するか確認
    if [[ -f "$file" ]]; then
        echo "Processing $file"
        evaluate_functional_correctness "$file"
        ((file_count++))
    fi
done

# 処理結果の出力
if [[ $file_count -eq 0 ]]; then
    echo "No .jsonl files found in the directory."
else
    echo "Total number of processed files: $file_count"
fi