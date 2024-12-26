#!/bin/bash

# 引数の解析
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    --load_generations_path)
      GENERATIONS_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 必須引数のチェック
if [ -z "$GENERATIONS_PATH" ]; then
    echo "Error: --load_generations_path is required"
    exit 1
fi

cd /work/gb20/b20042/model_merging
source env_mbpp/bin/activate

# 評価の実行
accelerate launch ./bigcode-evaluation-harness/main.py \
    --tasks mbpp \
    --allow_code_execution \
    --load_generations_path "$GENERATIONS_PATH" \
    --metric_output_path "evaluation_results.json"

# 評価結果からpass@1の値を抽出
ACCURACY=$(cat evaluation_results.json | jq -r '.mbpp."pass@1"')

# 結果を標準出力
echo "MBPP_ACCURACY=$ACCURACY"

