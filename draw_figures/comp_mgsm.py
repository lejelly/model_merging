import matplotlib
matplotlib.use("Agg")
import os
import json
import matplotlib.pyplot as plt

# データを読み込む関数
def load_data(file_path, is_ja):
    with open(file_path, 'r') as f:
        data = json.load(f)
    if is_ja:
        return [item for item in data if item['is_correct_ja']]
    else:
        return [item for item in data if item['is_correct_anylang']]

def draw_graph(paths, dir_name, is_ties, is_ja):
    # 各モデルのデータを読み込む
    modelA = load_data(paths['wizard'], is_ja)
    modelB = load_data(paths['shisa'], is_ja)
    if is_ties:
        modelC = load_data(paths['ties'], is_ja)
        modelD = load_data(paths['ties_dare'], is_ja)
    else:
        modelC = load_data(paths['task'], is_ja)
        modelD = load_data(paths['task_dare'], is_ja)

    # 図を作成
    fig, ax = plt.subplots(figsize=(12, 6))

    # 各モデルの正解をプロット
    ax.eventplot([item['problem_id'] for item in modelA], lineoffsets=3, linelengths=0.5, colors='green')
    ax.eventplot([item['problem_id'] for item in modelB], lineoffsets=2, linelengths=0.5, colors='orange')
    ax.eventplot([item['problem_id'] for item in modelC], lineoffsets=1, linelengths=0.5, colors='red')
    ax.eventplot([item['problem_id'] for item in modelD], lineoffsets=0, linelengths=0.5, colors='blue')

    # グラフの設定
    ax.set_yticks([0, 1, 2, 3])
    if is_ties:
        ax.set_yticklabels(['ties_merging(w/DARE)', 'ties_merging', 'shisa-gamma-7b-v1', 'WizardMath-7B-V1.1'])
    else:    
        ax.set_yticklabels(['task_arithmetic(w/DARE)', 'task_arithmetic', 'shisa-gamma-7b-v1', 'WizardMath-7B-V1.1'])
    ax.set_xlabel('Problem Id')
    ax.set_xlim(0, 250)
    if is_ja:
        ax.set_title('MGSM-JA (Response in Only Japanese)')
    else:
        ax.set_title('MGSM-JA (Response in Any Languages)')
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 図の保存
    if is_ja:
        fig_name = f'/work/gb20/b20042/model_merging/figs/mgsm-ja/{dir_name}/performance(acc-ja).png'
    else:
        fig_name = f'/work/gb20/b20042/model_merging/figs/mgsm-ja/{dir_name}/performance(acc).png'

    os.makedirs(os.path.dirname(fig_name), exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_name, dpi=200)

if __name__ == "__main__":   
    
    is_ties=False
    
    paths={
        'wizard' : '/work/gb20/b20042/model_merging/results_logging/single_model_inference/ja_mgsm/WizardLMTeam/WizardMath-7B-V1.1/withoutDARE_response.json',
        'shisa' : '/work/gb20/b20042/model_merging/results_logging/single_model_inference/ja_mgsm/augmxnt/shisa-gamma-7b-v1/withoutDARE_response.json',
        'task' : '/work/gb20/b20042/model_merging/results_logging/merged_model_inference/ja_mgsm/WizardMath-7B-V1.1_shisa-gamma-7b-v1/task_arithmetic/withoutDARE_response.json',
        'ties' : '/work/gb20/b20042/model_merging/results_logging/merged_model_inference/ja_mgsm/WizardMath-7B-V1.1_shisa-gamma-7b-v1/ties_merging/withoutDARE_response.json',
        'task_dare' : '/work/gb20/b20042/model_merging/results_logging/merged_model_inference/ja_mgsm/WizardMath-7B-V1.1_shisa-gamma-7b-v1/task_arithmetic/dare_response.json',
        'ties_dare' : '/work/gb20/b20042/model_merging/results_logging/merged_model_inference/ja_mgsm/WizardMath-7B-V1.1_shisa-gamma-7b-v1/ties_merging/dare_response.json'
    } 

    if is_ties:
        dir_name = 'comp_ties_dare'
    else:    
        dir_name = 'comp_taskarithmetic_dare'
    
    draw_graph(paths, dir_name, is_ties, is_ja=True)
    draw_graph(paths, dir_name, is_ties, is_ja=False)
    