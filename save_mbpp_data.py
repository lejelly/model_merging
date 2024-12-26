import json
from datasets import load_dataset

def save_filtered_dataset():
    # データセットの読み込み
    #ds = load_dataset("google-research-datasets/mbpp", "full")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # 保存先ディレクトリの作成
    import os
    os.makedirs("./math_code_data", exist_ok=True)
    
    # Task ID 601-974のデータを抽出して保存
    with open("./math_code_data/gsm8k.train.jsonl", "w", encoding="utf-8") as f:
        for item in dataset:  # datasetを直接イテレート
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

if __name__ == "__main__":
    save_filtered_dataset() 