# =========================================================
# script.py
#   - Accelerate + FSDP + λ学習 + 差分合成
# =========================================================
import os
import sys
import math
import json
import argparse
import logging
import random
import time
from typing import List, Dict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

# Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# Accelerate
from accelerate import Accelerator

# データ読込用
import jsonlines

from utils.load_config import cache_dir
from utils.utils import set_random_seed,smart_tokenizer_and_embedding_resize
from proposed_methods import WeightingStrategy

task_model_mapping_dict = {
    "math": "TIGER-Lab/MAmmoTH2-7B",
    "code": "Nondzu/Mistral-7B-codealpaca-lora",
    "jp": "augmxnt/shisa-gamma-7b-v1",
    
}
finetuned_model_backbone_mapping_dict = {
    "TIGER-Lab/MAmmoTH2-7B": "mistralai/Mistral-7B-v0.1",
    "Nondzu/Mistral-7B-codealpaca-lora": "mistralai/Mistral-7B-v0.1", 
    "augmxnt/shisa-gamma-7b-v1": "mistralai/Mistral-7B-v0.1",
}

# ---------------------------------------------------------
# 1) データ読み込み (GSM8K / MBPP / ja-MGSM) - ミニマル実装
# ---------------------------------------------------------
def load_gsm8k_data(path="math_code_data/gsm8k.train.jsonl", num_samples=10, seed=42):
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for item in jsonlines.Reader(f):
                data.append({"question": item["question"], "answer": item["answer"]})
    except:
        pass
    random.seed(seed)
    random.shuffle(data)
    return data[:num_samples]

def load_mbpp_data(path="math_code_data/mbpp.train.jsonl", num_samples=10, seed=42):
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for item in jsonlines.Reader(f):
                data.append({"text": item["text"], "code": item["code"]})
    except:
        pass
    random.seed(seed)
    random.shuffle(data)
    return data[:num_samples]

def load_ja_mgsm_data(path="math_code_data/ja_mgsm.train.jsonl", num_samples=10, seed=42):
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for item in jsonlines.Reader(f):
                data.append({"question": item["question"], "answer": item["answer"]})
    except:
        pass
    random.seed(seed)
    random.shuffle(data)
    return data[:num_samples]

# ---------------------------------------------------------
# 2) Task Vector (ベースモデルとfinetunedモデルのパラメータ差分)
# ---------------------------------------------------------
class TaskVector:
    """
    pretrained_model と finetuned_model のパラメータ差分を保存
    """
    def __init__(self, base_model: PreTrainedModel, ft_model: PreTrainedModel):
        self.diff = {}
        with torch.no_grad():
            for (name, p_base), (_, p_ft) in zip(base_model.named_parameters(), ft_model.named_parameters()):
                # CPU上で計算する（メモリ節約のため half 変換）
                d = (p_ft.cpu() - p_base.cpu()).half()
                self.diff[name] = d

# ---------------------------------------------------------
# 3) 差分合成モデル (nn.Module)
#    forward()時に: param = base_param + sum(lambda_i * diff_i)
# ---------------------------------------------------------
class MergedModelForCausalLM(nn.Module):
    """
    - ベースモデルのパラメータを .base_params にBufferとして保存
    - 複数の差分(TaskVector)を .task_diffs に保持
    - λは nn.Parameter(...) で管理 (requires_grad=True)
    """
    def __init__(
        self,
        base_model: PreTrainedModel,
        task_vectors: List[TaskVector],
        init_lambdas: np.ndarray = None
    ):
        super().__init__()
        # ベースモデルをコピーして保持
        # ただし parameter ではなく、バッファ (requires_grad=False) として保存する
        self.base_names = []
        for name, param in base_model.named_parameters():
            # nn.Buffer として登録
            # 「self.register_buffer(name, data, persistent=False)」の方がよいが
            # name衝突を避けるために辞書にまとめる
            pass
        self.base_params = nn.ParameterDict()  # ここを ParameterDict ではなく BufferDict などにしてもOK
        with torch.no_grad():
            for name, p in base_model.named_parameters():
                # base_paramは学習しない → requires_grad=False
                # ただし FSDP環境でバッファにするとリサイズ時の対応が難しいため
                # ParameterDict に入れるが勾配は止めておく
                clone_p = p.data.clone().half().cpu()  # CPU上で保持
                # ここでは一旦 half().cpu() へ転送して保存
                # FSDP で使うなら gpu() かつ flatten() される可能性があるので本当は注意が必要
                param_ = nn.Parameter(clone_p, requires_grad=False)
                self.base_params[name.replace('.', '_')] = param_
                self.base_names.append(name)

        # TaskVectorを格納
        self.task_diffs = []
        for tv in task_vectors:
            self.task_diffs.append(tv.diff)

        # λ
        if init_lambdas is None:
            init_lambdas = np.zeros(len(task_vectors), dtype=np.float16)
        self.lambdas = nn.Parameter(torch.tensor(init_lambdas, dtype=torch.float16), requires_grad=True)

        # 実際の推論などで使う 変換用に base_model の config /  tokenizer.pad_token_id など流用する
        # forward時に base_modelのtransformerモジュールを使いたいが、ここでは独自に定義
        # → あるいは "self.wrapped_model = base_model" として普通に持つ手もあるが
        #    "パラメータを再初期化しない" ように注意。transformerブロックだけサブクラス化するなど要工夫。
        #
        # 今回はシンプルに "self.base_model_for_forward" を作っておくが、
        # そちらのparameters() は未使用にする (学習対象は lam + diff なので)
        self.base_model_for_forward = base_model

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor = None,
        **kwargs
    ):
        """
        1. base_params + Σ( λ_i * diff_i ) を計算して、パラメータを一時的に更新
        2. その updated_params で base_model_for_forward を forward
        3. ロス or logits を返す
        """
        # ここでは「パラメータを実際に上書き→推論→元に戻す」処理を実装
        # (stateless.functional_callを使わない方針)

        device = input_ids.device
        # λ (GPU)
        lambdas_ = self.lambdas.to(device)

        # base_model_for_forward のパラメータを "manual override" する
        # → 大規模モデルだと毎回コピーでメモリ大になるので、本番環境では工夫が必要
        with torch.no_grad():
            for name, base_tensor in self.base_params.items():
                # name は '.' が '_' に置換された形
                param_name = name.replace('_', '.')
                merged = base_tensor.half().to(device).clone()  # float16

                # 各タスクベクトルを足し合わせ
                for i, diff_dict in enumerate(self.task_diffs):
                    diff_cpu = diff_dict[param_name]  # shapeは同じ
                    merged += (lambdas_[i] * diff_cpu.half().to(device))

                # base_model_for_forward 内の param_name パラメータを上書き
                # find param pointer:
                p_ = dict(self.base_model_for_forward.named_parameters())[param_name]
                p_.data = merged.half()

        # forward
        outputs = self.base_model_for_forward(
            input_ids=input_ids,
            labels=labels,
            **kwargs
        )
        return outputs

# ---------------------------------------------------------
# 4) 学習ロジック: 3つのタスク (GSM8K / MBPP / jaMGSM) の CrossEntropy ロスを合計
# ---------------------------------------------------------
class LambdaTrainer:
    def __init__(
        self,
        merged_model: MergedModelForCausalLM,
        tokenizer_list: List[AutoTokenizer],
        args,
    ):
        self.accelerator = Accelerator()  # FSDP 有効化は accelerate config で設定
        self.merged_model = merged_model
        self.args = args

        # トークナイザ (3種類)
        self.tok_gsm8k, self.tok_mbpp, self.tok_jamgsm = tokenizer_list

        # ローダ
        self.gsm8k_data = load_gsm8k_data(num_samples=args.num_train_samples, seed=args.seed)
        self.mbpp_data = load_mbpp_data(num_samples=args.num_train_samples, seed=args.seed)
        self.jamgsm_data = load_ja_mgsm_data(num_samples=args.num_train_samples, seed=args.seed)

        # Optimizer は λ のみ学習対象
        self.optimizer = torch.optim.AdamW([p for p in merged_model.parameters() if p.requires_grad], lr=args.learning_rate)

        # モデルを FSDPラップ or DDPラップ
        self.merged_model = self.accelerator.prepare(self.merged_model)
        self.optimizer = self.accelerator.prepare(self.optimizer)

        # シード
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    def make_batches(self, data_list, bsz):
        return [data_list[i:i+bsz] for i in range(0, len(data_list), bsz)]

    def encode_prompt_and_label(
        self,
        prompt_texts: List[str],
        answer_texts: List[str],
        tokenizer: AutoTokenizer,
        max_length=256
    ):
        """
        "プロンプト部分" は -100 でマスクし、"回答部分" だけが学習対象になるようにする。
        """
        device = self.accelerator.device
        pad_id = tokenizer.pad_token_id or 0

        enc_prompt = tokenizer(prompt_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc_answer = tokenizer(answer_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

        input_ids_list = []
        labels_list = []

        for i in range(len(prompt_texts)):
            p_ids = enc_prompt.input_ids[i].tolist()
            a_ids = enc_answer.input_ids[i].tolist()

            # paddingを除去して長さを計測
            p_len = sum(1 for x in p_ids if x != pad_id)
            a_len = sum(1 for x in a_ids if x != pad_id)

            merged_ids = p_ids[:p_len] + a_ids[:a_len]
            merged_labels = ([-100] * p_len) + a_ids[:a_len]

            input_ids_list.append(merged_ids)
            labels_list.append(merged_labels)

        # バッチ内最大長でパディング
        max_len = max(len(ids) for ids in input_ids_list)
        for i in range(len(input_ids_list)):
            diff_ = max_len - len(input_ids_list[i])
            input_ids_list[i].extend([pad_id]*diff_)
            labels_list[i].extend([-100]*diff_)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels = torch.tensor(labels_list, dtype=torch.long, device=device)
        return input_ids, labels

    def compute_gsm8k_loss(self, batch_data):
        print("Computing GSM8K loss...", flush=True)
        prompts = [f"Q: {d['question']}\nA:" for d in batch_data]
        answers = [d['answer'] for d in batch_data]
        input_ids, labels = self.encode_prompt_and_label(prompts, answers, self.tok_gsm8k)
        outputs = self.merged_model(input_ids=input_ids, labels=labels)
        return outputs.loss

    def compute_mbpp_loss(self, batch_data):
        print("Computing MBPP loss...", flush=True)
        prompts = [f"{d['text']}\nSolution:\n" for d in batch_data]
        answers = [d['code'] for d in batch_data]
        input_ids, labels = self.encode_prompt_and_label(prompts, answers, self.tok_mbpp)
        outputs = self.merged_model(input_ids=input_ids, labels=labels)
        return outputs.loss

    def compute_jamgsm_loss(self, batch_data):
        print("Computing jaMGSM loss...", flush=True)
        prompts = [f"{d['question']}\n答えを考える: " for d in batch_data]
        answers = [d['answer'] for d in batch_data]
        input_ids, labels = self.encode_prompt_and_label(prompts, answers, self.tok_jamgsm)
        outputs = self.merged_model(input_ids=input_ids, labels=labels)
        return outputs.loss

    def train_one_epoch(self, epoch_idx):
        print(f"\nStarting epoch {epoch_idx}/{self.args.num_epochs-1}...", flush=True)
        self.merged_model.train()
        batch_size = self.args.batch_size

        # 3タスクと、それぞれのロス計算関数をまとめる
        tasks = [
            ("gsm8k",   self.gsm8k_data,   self.compute_gsm8k_loss),
            ("mbpp",    self.mbpp_data,    self.compute_mbpp_loss),
            ("ja_mgsm", self.jamgsm_data,  self.compute_jamgsm_loss),
        ]

        # 合計ロス (勾配を流すために requires_grad=True にしておく)
        # ただし .to(self.accelerator.device) しておくか、あとでロス計算時にGPU上に置かれるのであれば不要でも可
        total_loss = torch.tensor(0.0, device=self.accelerator.device, requires_grad=True)

        # 各タスクのロスを足し合わせる
        for task_name, dataset, loss_fn in tasks:
            # データをバッチに分割
            batches = self.make_batches(dataset, batch_size)

            # 平均ロス確認用
            task_loss_sum = 0.0

            # バッチごとにロスを計算して合計
            for batch in batches:
                loss = loss_fn(batch)  # 例: self.compute_gsm8k_loss(batch)
                total_loss = total_loss + loss
                task_loss_sum += loss.item()

            # タスク平均ロス (ログ出力用)
            avg_task_loss = task_loss_sum / max(len(batches), 1)
            if self.accelerator.is_main_process:
                print(f"[{task_name}] loss = {avg_task_loss:.4f}")

        # ここで一度だけ backward して勾配更新
        self.accelerator.backward(total_loss)
        self.optimizer.step()
        self.optimizer.zero_grad()

        # ログ出力
        total_loss_val = total_loss.item()
        if self.accelerator.is_main_process:
            print(f"[Epoch {epoch_idx}] total_loss = {total_loss_val:.4f}")
            print(f"Lambda values = {self.merged_model.lambdas.detach().cpu().numpy()}")

            # CSV保存
            if args.optimized_lambda_filepath:
                optimized_df = pd.DataFrame({
                    'epoch': [epoch_idx + 1],
                    'model_name': [args.finetuned_model_names],
                    'lambdas': [self.merged_model.lambdas.detach().cpu().numpy().tolist()],
                    'loss': [total_loss_val]
                })
                if os.path.exists(args.optimized_lambda_filepath):
                    optimized_df.to_csv(args.optimized_lambda_filepath, mode='a', header=False, index=False)
                else:
                    optimized_df.to_csv(args.optimized_lambda_filepath, index=False)

        return total_loss_val

    def train(self):
        print("Starting training...", flush=True)
        best_loss = 1e9
        best_lam = None
        for epoch in range(self.args.num_epochs):
            print(f"\n=== Epoch {epoch}/{self.args.num_epochs-1} ===", flush=True)
            loss_val = self.train_one_epoch(epoch)
            # 最良更新
            if loss_val < best_loss:
                best_loss = loss_val
                best_lam = self.merged_model.lambdas.detach().cpu().numpy().copy()
                print(f"New best loss: {best_loss:.4f}", flush=True)

        if self.accelerator.is_main_process:
            print(f"Training Done. Best loss={best_loss:.4f}")
            print(f"Best Lambdas = {best_lam}")

# ---------------------------------------------------------
# 5) main: 実行スクリプト (Accelerateを使いFSDPで学習)
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train_samples", type=int, default=10, help="Train samples per task for quick debug")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="lr for lambdas")
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial_lambda_filepath", type=str, default=None, help="initial lambda filepath")
    parser.add_argument("--optimized_lambda_filepath", type=str, default=None, help="optimized lambda filepath")
    parser.add_argument("--lambda_strategy", type=str, default=None, help="lambda strategy")
    args = parser.parse_args()
    
    set_random_seed(seed=args.seed)
    
    # 例: mergeフラグ→ finetuned_model_names の決定
    finetuned_model_names = []
    finetuned_model_names.append(task_model_mapping_dict["math"])
    finetuned_model_names.append(task_model_mapping_dict["code"])
    finetuned_model_names.append(task_model_mapping_dict["jp"])
    args.finetuned_model_names = finetuned_model_names

    # ファイル名の生成
    strategy = WeightingStrategy(args.lambda_strategy)
    params_name = f"{strategy.value}/adamW_epochs{args.num_epochs}_lr{args.learning_rate}_sample{args.num_train_samples}"
    save_dir = f"./lambdas/seed{args.seed}/math_code_jp/{params_name}" 
    os.makedirs(save_dir, exist_ok=True)
    model_names_str = '_'.join(sorted([name.split('/')[-1] for name in finetuned_model_names]))
    initial_lambda_filename = f'initial_lambdas_{strategy.value}_{model_names_str}.csv'
    optimized_lambda_filename = f'optimized_lambdas_{strategy.value}_{model_names_str}.csv'

    #　λの初期化
    if args.initial_lambda_filepath is not None:
        initial_lambda_filepath = args.initial_lambda_filepath
    else:
        initial_lambda_filepath = os.path.join(save_dir, initial_lambda_filename)
    
    if args.optimized_lambda_filepath is not None:
        optimized_lambda_filepath = args.optimized_lambda_filepath
    else:
        args.optimized_lambda_filepath = os.path.join(save_dir, optimized_lambda_filename)
    
    # initial lambdasの読み込み
    if os.path.exists(initial_lambda_filepath):
        print(f"Loading existing lambdas from: {initial_lambda_filepath}")
        initial_df = pd.read_csv(initial_lambda_filepath)
        initial_df = initial_df.set_index('model_name').loc[finetuned_model_names].reset_index()
        initial_lambdas = initial_df['lambda'].values
    else:    
        # 新しくλ値を計算
        if args.lambda_strategy == "metagpt_random" or args.lambda_strategy == "optuna":
            # ランダムな値(0-1)を生成
            initial_lambdas = np.random.rand(len(models_to_merge))
        elif args.lambda_strategy == "metagpt_random_normalize":
            # ランダムな値を生成し、合計が1になるように正規化
            random_values = np.random.rand(len(models_to_merge))
            initial_lambdas = random_values / np.sum(random_values)
        elif args.lambda_strategy == "metagpt_average":
            # モデル数で均等に分割
            initial_lambdas = np.array([1.0 / len(models_to_merge)] * len(models_to_merge))
        elif args.lambda_strategy == "metagpt_all_one":
            # すべての要素を1.0に設定
            initial_lambdas = np.array([1.0] * len(models_to_merge))
            
        # initial lambdasの保存
        if args.lambda_strategy != "optuna":
            initial_df = pd.DataFrame({
                'model_name': finetuned_model_names,
                'lambda': initial_lambdas
            })
            initial_df.to_csv(initial_lambda_filepath, index=False)

        

    logging.basicConfig(level=logging.INFO)

    # シード
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    pretrained_model_names = [finetuned_model_backbone_mapping_dict[finetuned_model_name] for finetuned_model_name in finetuned_model_names]
    args.pretrained_model_name = pretrained_model_names[0]

    # 3. ベースモデル読み込み (CPU上; bfloat16 例)
    base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name),device_map="cpu",torch_dtype=torch.bfloat16)
    base_model_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name))
    base_model.eval()
    
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=dict(pad_token="[PAD]"),
        model=base_model,
        tokenizer=base_model_tokenizer,
    )

    # 5. 差分取り出し: TaskVector を作る
    task_vectors = []
    finetuned_tokenizers = []
    for ft_name in finetuned_model_names:
        # モデルを1つずつロードして差分を計算し、すぐに解放
        ft_model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, ft_name),
            device_map="cpu",
            torch_dtype=torch.bfloat16
        )
        ft_model.eval()
        
        ft_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=os.path.join(cache_dir, ft_name)
        )
        finetuned_tokenizers.append(ft_tokenizer)
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=ft_model,
            tokenizer=ft_tokenizer,
        )
        
        # 差分を計算
        tv = TaskVector(base_model, ft_model)
        task_vectors.append(tv)
        
        # モデルを明示的に解放
        del ft_model
        torch.cuda.empty_cache()  # GPU メモリもクリア

    # 6. 合成モデルを作成
    init_lambdas = np.zeros(len(task_vectors), dtype=np.float32)
    merged_model = MergedModelForCausalLM(base_model, task_vectors, init_lambdas)

    # 7. LambdaTrainer (3つのタスク = gsm8k, mbpp, ja_mgsm のロス合計例)
    #    ここで "finetuned_tokenizers" のうち「どれを gsm8k / mbpp / ja_mgsm に使うか」は
    #    ユースケースに応じて決めてください。(例: 先頭3個を流用など)
    trainer = LambdaTrainer(
        merged_model=merged_model,
        tokenizers=finetuned_tokenizers[:3],  # 例: 先頭3個 (実際はお好みで)
        args=args
    )

    # 8. 学習を実行
    trainer.train()


if __name__ == "__main__":
    # accelerate launch script.py (...args...)
    main()
