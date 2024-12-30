#!/bin/bash

cd /work/gb20/b20042/model_merging
source env_mbpp/bin/activate

# jsonlファイルを一つずつ処理
for file in /work/gb20/b20042/model_merging/save_gen_codes_results/math_code_jp/mbpp/*.jsonl; do
    # ファイル名（拡張子なし）を取得
    filename=$(basename "$file" .jsonl)
    
    # コマンドを実行し、結果をファイルに出力
    accelerate launch ./bigcode-evaluation-harness/main.py \
        --tasks mbpp \
        --allow_code_execution \
        --load_generations_path "$file" \
        --metric_output_path "/work/gb20/b20042/model_merging/save_gen_codes_results/math_code_jp/mbpp/results/${filename}.txt"
    
    echo "処理完了: $filename"
done
