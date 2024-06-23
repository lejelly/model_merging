import matplotlib.pyplot as plt
import numpy as np
import os
import ast
from tqdm import tqdm
import datasets
import random
import torch
import glob
import re

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# MGSM-JAの全問題を読み込む関数
def load_mgsm_ja_problems():
    eval_set = datasets.load_dataset(path="juletxara/mgsm", split="test", name="ja")
    question = [example["question"] for example in eval_set]
    return question

def get_all_log_files(root_dir):
    # root_dirを絶対パスに変換
    root_dir = os.path.abspath(root_dir)
    
    # 全てのlogファイルのパスを取得
    log_files = glob.glob(os.path.join(root_dir, '**', '*.log'), recursive=True)
    
    # 絶対パスのリストを返す
    return [os.path.abspath(file) for file in log_files]

def extract_model_name(path):
    # 正規表現パターン
    pattern = r'/([^/]+)_inference_mask'
    
    # パターンにマッチする部分を検索
    match = re.search(pattern, path)
    
    if match:
        return match.group(1)
    else:
        return None

# モデルが間違えた問題を読み込む関数
def load_model_a_mistakes(file_path, is_ja):
    # ログファイルを読み込む
    with open(file_path, 'r', encoding='utf-8') as f:
        log_content = f.read()
    
    # invalid_outputs セクションを見つける
    if is_ja:
        key = "acc-ja: invalid outputs"
        start_index = log_content.find(key)
    else:   
        key = "acc: invalid outputs"
        key_acc_ja = "acc-ja: invalid outputs"
        start_index = log_content.find(key)
        start_index_acc_ja = log_content.find(key_acc_ja)
            
    if start_index != -1:
        if is_ja:
            invalid_outputs = log_content[start_index + len(key):]
        else:
            invalid_outputs = log_content[start_index + len(key):start_index_acc_ja]
        first_bracket = invalid_outputs.find('[')
        last_bracket = invalid_outputs.rfind(']')

        if first_bracket != -1 and last_bracket != -1:
            invalid_outputs = invalid_outputs[first_bracket:last_bracket+1]
            invalid_outputs = ast.literal_eval(invalid_outputs)
            # 各問題から 'question' フィールドを抽出
            questions = [item['question'] for item in invalid_outputs]
            return questions
    else:
        print("invalid_outputs セクションが見つかりませんでした。")
    
    return []

def create_mistake_array(all_problems, mistake_problems):
    # 全ての問題に対して1を持つ配列を初期化
    mistake_array = np.ones(len(all_problems), dtype=int)
    
    # mistake_problemsの各問題に対して
    for problem in mistake_problems:
        # all_problemsの中でその問題のインデックスを見つける
        if problem in all_problems:
            index = all_problems.index(problem)
            # そのインデックスの要素を0に設定
            mistake_array[index] = 0
    
    return mistake_array

def draw_graph(all_problems, paths, is_ja):
    model_names = []
    mistake_problems = []
    for file_path in paths:
        model_names.append(extract_model_name(file_path))
        mistake_problems.append(load_model_a_mistakes(file_path, is_ja))
    
    data = []
    for mitake in mistake_problems:
        data.append(create_mistake_array(all_problems, mitake))
        
    # データの数
    num_problems = len(all_problems)

    # ダミーデータの生成（0: 不正解, 1: 正解）
    data = np.array(data)

    # プロットの作成
    fig, ax = plt.subplots(figsize=(15, 6))

    # 各モデルのデータをプロット
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    for i, model in enumerate(model_names):
        correct_answers = np.where(data[i] == 1)[0]
        ax.scatter(correct_answers, [i] * len(correct_answers), color=colors[i], marker='|', s=100)

    # グラフの設定
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names)
    ax.set_xlabel('Problem Id')
    ax.set_xlim(0, num_problems)
    if is_ja:
        ax.set_title('MGSM-JA (Only Japanese Responses are acceptable.)')
    else:
        ax.set_title('MGSM-JA (English responses are also acceptable.)')

    # グリッド線の追加
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    # 図の保存
    if is_ja:
        fig_name = f'/work/gb20/b20042/model_merging/figs/performance_overview(acc-ja).png'
    else:
        fig_name = f'/work/gb20/b20042/model_merging/figs/performance_overview(acc).png'
        
    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=300)
    plt.show()


if __name__ == "__main__":
    set_random_seed(seed=0)
    save_logs_path = '/work/gb20/b20042/model_merging/save_logs'    
    all_problems = load_mgsm_ja_problems()
    paths = get_all_log_files(save_logs_path)

    draw_graph(all_problems, paths, is_ja=True)
    draw_graph(all_problems, paths, is_ja=False)