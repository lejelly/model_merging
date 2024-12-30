import pandas as pd
import numpy as np

# データの読み込み
df1 = pd.read_csv('1234.csv')
df2 = pd.read_csv('3232.csv')
df3 = pd.read_csv('73794.csv')

# 各手法のデータを集める
methods = ['random', 'all one', 'average', 'optuna', 'metagpt_optimize']
metrics = ['gsm8k', 'mbpp', 'ja_mgsm']

results = {}
for method in methods:
    results[method] = {
        'gsm8k': [],
        'mbpp': [],
        'ja_mgsm': []
    }
    for df in [df1, df2, df3]:
        row = df[df['name'] == method].iloc[0]
        for metric in metrics:
            if pd.notna(row[metric]):  # NaN値をスキップ
                results[method][metric].append(row[metric])

# 平均と分散を計算
summary = {}
for method in methods:
    summary[method] = {
        metric: {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1) if len(values) > 1 else 0
        }
        for metric, values in results[method].items()
        if values  # 空のリストをスキップ
    }

# 結果の表示
for method in methods:
    print(f"\n{method}:")
    for metric in metrics:
        if metric in summary[method]:
            mean = summary[method][metric]['mean']
            std = summary[method][metric]['std']
            print(f"{metric}: {mean:.4f} ± {std:.4f}")