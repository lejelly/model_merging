#!/bin/bash

# カレントディレクトリを設定
cd /work/gb20/b20042/model_merging

# test.XXX.outというファイルを取得
for file in test.*.out; do
  # 最後の行を取得
  last_line=$(tail -n 1 "$file")
  
  # 最後の行に"completed"が含まれているかチェック
  if [[ "$last_line" != *"completed"* ]]; then
    # "completed"が含まれていない場合、そのファイル名をerro.txtに追記
    echo "$file" >> erro.txt
  fi
done