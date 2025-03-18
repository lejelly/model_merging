from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.evaluate_llms_utils import extract_answer_number, get_math_task_prompt
from tqdm import tqdm

# 1) モデルとトークナイザのロード
model_name = "mistralai/Mistral-7B-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",   # GPU分散など環境に応じて指定
    torch_dtype=torch.float16,  # 16bitにしたい場合など
)

# 2) GSM8Kのテストセットをロード
#    'main' スプリットがtrain, 'test'スプリットが別にある。元データに注意。
dataset = load_dataset("gsm8k", "main")
test_data = dataset["test"]

# 3) テストサンプルを一定数ループ (ここでは簡単のため先頭の数件だけなど)
num_samples_to_eval = 50 # len(test_data)  # 実際には len(test_data) など
results = []

# プロンプトテンプレートを取得（デフォルトはzeroshotcot）
problem_prompt = get_math_task_prompt(prompt_type="zeroshotcot")

print(f"評価サンプル数: {num_samples_to_eval}")
for idx in tqdm(range(num_samples_to_eval), desc="サンプル評価中"):
    sample = test_data[idx]
    question = sample["question"]
    gold_answer = sample["answer"]  # GSM8Kでは解答文字列が入っている
    
    #print(f"question: {question}")
    #print(f"gold_answer: {gold_answer}")

    # 入力プロンプトを作る
    prompt_text = problem_prompt.format(instruction=question)

    # トークナイズ
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    # 4) モデルに生成させる: max_new_tokens, temperature などは適宜調整
    with torch.no_grad():
        generation_output = model.generate(
            input_ids,
            max_new_tokens=1024,
            output_scores=True,            # スコア(=logits)を返してもらう
            return_dict_in_generate=True,  # 生成プロセス情報を辞書で受け取る
        )
    
    # 生成結果テキスト
    gen_ids = generation_output.sequences[0]
    gen_text = tokenizer.decode(gen_ids[len(input_ids[0]):], skip_special_tokens=True)

    # 生成ステップごとのscores (List of tensors)
    # それぞれ [batch_size, vocab_size] 形状の logits
    all_scores = generation_output.scores

    # 5) 各ステップのエントロピーを計算
    step_entropies = []
    for step_idx, logits in enumerate(all_scores):
        # softmaxで確率化 (batch_size=1想定)
        probs = torch.softmax(logits[0], dim=-1)
        # エントロピー計算 = - \sum p*log(p)
        # logは自然対数なので2進や10進にする場合は適宜変更
        entropy = - torch.sum(probs * torch.log(probs + 1e-9))
        step_entropies.append(entropy.item())

    # 生成ステップ中の平均エントロピー
    avg_entropy = float(np.mean(step_entropies))

    # 6) 正解判定
    #   GSM8Kの回答フォーマットは色々なので、extract_answer_number関数を使用して
    #   生成されたテキストから数値を抽出し、正解と比較する
    y_pred = extract_answer_number(gen_text)
    if y_pred is not None:
        # gold_answerから数値を抽出（GSM8Kの場合、"#### 数値"の形式）
        gold_number = int(gold_answer.split('#### ')[-1].replace(',', ''))
        is_correct = 1 if y_pred == gold_number else 0
    else:
        is_correct = 0

    results.append({"entropy": avg_entropy, "correct": is_correct})
    
    #print(f"y_pred: {y_pred}, gold_number: {gold_number}, is_correct: {is_correct}")

# 7) 結果をエントロピーの大きさでビニング → 正解率算出
#    bins = [0.0, 0.1, 0.2, ..., 1.0, float("inf")] など任意設定
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")]
bin_labels = []
bin_correct_counts = [0]*(len(bins)-1)
bin_total_counts   = [0]*(len(bins)-1)

for r in results:
    e = r["entropy"]
    c = r["correct"]
    # どのbinに属するか判定
    for b_i in range(len(bins)-1):
        if bins[b_i] <= e < bins[b_i+1]:
            bin_correct_counts[b_i] += c
            bin_total_counts[b_i] += 1
            break

# binごとの正解率( accuracy )
bin_accs = []
for b_i in range(len(bins)-1):
    # (正解数 / サンプル総数) ただしサンプル0の場合は0など
    acc = (
        bin_correct_counts[b_i]/bin_total_counts[b_i]
        if bin_total_counts[b_i] > 0
        else 0.0
    )
    bin_accs.append(acc)

# binのラベル(文字列)を作る例
bin_labels = []
for b_i in range(len(bins)-1):
    left = bins[b_i]
    right = bins[b_i+1]
    if right == float("inf"):
        bin_labels.append(f">={left:.1f}")
    else:
        bin_labels.append(f"({left:.1f},{right:.1f}]")

# 8) 棒グラフを描画
plt.figure(figsize=(8,4))
plt.bar(range(len(bin_accs)), bin_accs, width=0.6)
plt.xticks(range(len(bin_accs)), bin_labels, rotation=45)
plt.xlabel("Entropy bins")
plt.ylabel("Accuracy")
plt.title("Entropy vs Accuracy (Mistral-7B on GSM8K, sample)")
plt.ylim([0,1])
plt.tight_layout()
plt.savefig('entropy_vs_accuracy.png', dpi=300, bbox_inches='tight')
plt.show()
