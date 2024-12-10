
#!/bin/bash

cd /work/gb20/b20042/model_merging
source env_eval/bin/activate

# ディレクトリパスを設定
TARGET_DIR="/work/gb20/b20042/model_merging/save_gen_codes_results/math_code_jp/human_eval"

# エラーハンドリング関数
handle_error() {
    echo "エラーが発生しました: $1"
    exit 1
}

# ディレクトリの存在確認
if [ ! -d "$TARGET_DIR" ]; then
    handle_error "指定されたディレクトリが存在しません: $TARGET_DIR"
fi

# カレントディレクトリを変更
cd "$TARGET_DIR" || handle_error "ディレクトリの変更に失敗しました"

# 処理したファイル数をカウント
count=0

# すべての.jsonlファイルを処理
for file in *.jsonl; do
    # ファイルが実際に存在することを確認
    if [ -f "$file" ]; then
        echo "処理中のファイル: $file"
        
        # pjobコマンドを実行
        evaluate_functional_correctness  "$file" || handle_error "pjobコマンドの実行に失敗しました: $file"
        
        # カウンターをインクリメント
        ((count++))
    fi
done

# 結果を表示
echo "処理完了"
echo "処理したファイル数: $count"

# 処理したファイルが0の場合は警告を表示
if [ $count -eq 0 ]; then
    echo "警告: .jsonlファイルが見つかりませんでした"
fi

