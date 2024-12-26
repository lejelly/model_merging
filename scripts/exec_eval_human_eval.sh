
#!/bin/bash

cd /work/gb20/b20042/model_merging
source env_eval/bin/activate

# ディレクトリパスを設定
TARGET_DIR="PATH_TO_SAVE_GEN_CODES_RESULTS/human_eval"
file="PATH_TO_SAVE_GEN_CODES_RESULTS/human_eval/XXX.jsonl"

evaluate_functional_correctness  "$file" || handle_error "pjobコマンドの実行に失敗しました: $file"

