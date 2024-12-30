import os
import re
import pandas as pd
import json

def extract_accuracy(file_path, is_gsm8k=True):
    with open(file_path, 'r') as f:
        content = f.read().strip()
        if is_gsm8k:
            match = re.search(r'accuracy: ([\d.]+)', content)
        else:
            match = re.search(r'accuracy_ja: ([\d.]+)', content)
        if match:
            return float(match.group(1))
    return None

def extract_mbpp_accuracy(file_path):
    with open(file_path, 'r') as f:
        content = json.load(f)
        return content['mbpp']['pass@1']

def process_seed(seed_number):
    base_dir = f'/work/gb20/b20042/model_merging/results_metagpt/{seed_number}/math_code_jp'
    # 結果を格納する辞書
    results = {
        'random': {'name': 'random', 'gsm8k': None, 'mbpp': None, 'ja_mgsm': None},
        'all_one': {'name': 'all one', 'gsm8k': None, 'mbpp': None, 'ja_mgsm': None},
        'average': {'name': 'average', 'gsm8k': None, 'mbpp': None, 'ja_mgsm': None},
        'optuna': {'name': 'optuna', 'gsm8k': None, 'mbpp': None, 'ja_mgsm': None},
        'optimize': {'name': 'metagpt_optimize', 'gsm8k': None, 'mbpp': None, 'ja_mgsm': None}
    }
    
    # ディレクトリマッピング
    dir_mapping = {
        'metagpt_random': 'random',
        'metagpt_all_one': 'all_one',
        'metagpt_average': 'average',
        'optuna': 'optuna',
        'metagpt_optimize': 'optimize'
    }
    
    # MBPPスコアのベースディレクトリ
    mbpp_base_dir = '/work/gb20/b20042/model_merging/save_gen_codes_results/math_code_jp/mbpp/results'
    
    # 各ディレクトリを処理
    for dir_name in dir_mapping.keys():
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.exists(dir_path):
            # GSM8Kスコアを取得
            gsm8k_path = os.path.join(dir_path, 'gsm8k.txt')
            if os.path.exists(gsm8k_path):
                results[dir_mapping[dir_name]]['gsm8k'] = extract_accuracy(gsm8k_path, True)
            
            # JA-MGSMスコアを取得
            ja_mgsm_path = os.path.join(dir_path, 'ja_mgsm.txt')
            if os.path.exists(ja_mgsm_path):
                results[dir_mapping[dir_name]]['ja_mgsm'] = extract_accuracy(ja_mgsm_path, False)
            
            # MBPPスコアを取得
            mbpp_filename = f'seed{seed_number}_{dir_name}_adam_epochs20_lr0.0001_sample10.txt'
            mbpp_path = os.path.join(mbpp_base_dir, mbpp_filename)
            if os.path.exists(mbpp_path):
                results[dir_mapping[dir_name]]['mbpp'] = extract_mbpp_accuracy(mbpp_path)
    
    # DataFrameを作成して保存
    df = pd.DataFrame([v for v in results.values()])
    output_path = f'/work/gb20/b20042/model_merging/results_for_thesis/{seed_number}.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

# 全てのシードに対して処理を実行
seeds = [1234, 3232, 73794]
for seed in seeds:
    process_seed(seed)
