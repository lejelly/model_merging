import torch
import torch.nn.functional as F
import json
import logging
import random
import os
import wandb
import pandas as pd
import jsonlines
from typing import List, Dict, Optional
import numpy as np
import copy
import optuna
import torch.nn as nn
import psutil
import time
import gc
from tqdm import tqdm
from torch.nn.utils import stateless

# Mixed Precision (任意)
from torch.cuda.amp import autocast, GradScaler

# Cosine Warmupスケジューラーのヘルパー関数
import math

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


###############################################################################
# 以下はサンプルの読み込み関数 (データセット)
###############################################################################
def load_gsm8k_data(path="math_code_data/gsm8k.train.jsonl", num_samples=10, seed=42):
    data = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for item in jsonlines.Reader(f):
                data.append({
                    "question": item["question"],
                    "answer": item["answer"]
                })
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
                data.append({
                    "text": item["text"],
                    "code": item["code"]
                })
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
                data.append({
                    "question": item["question"],
                    "answer": item["answer"],
                    "answer_number": item["answer_number"]
                })
    except:
        pass
    random.seed(seed)
    random.shuffle(data)
    return data[:num_samples]


###############################################################################
# 1) 差分をGPU上に保持するクラス (変更なし)
###############################################################################
class FastTaskVector:
    def __init__(self, pretrained_model: nn.Module, finetuned_model: nn.Module):
        self.delta_dict = {}
        
        with torch.no_grad():
            # 両方のモデルをCPUに移動
            pretrained_model = pretrained_model
            finetuned_model = finetuned_model
            
            # 全パラメータに対して一度に差分を計算
            for (name_pre, p_pre), (name_ft, p_ft) in tqdm(
                zip(pretrained_model.named_parameters(), finetuned_model.named_parameters()),
                desc="Calculating parameter deltas"
            ):
                assert name_pre == name_ft
                # CPU上で差分を計算してhalf精度に変換
                self.delta_dict[name_pre] = (p_ft - p_pre).half()
            

###############################################################################
# 2) forward 時にレイヤーごとに合成するラッパークラス (修正ポイント)
###############################################################################
class LayerwiseMergedModelWrapper(nn.Module):
    """
    - merge() で「base_param + λ*delta」を一度だけ計算し、self.merged_params に保持
    - forward() は、self.merged_params を使って functional_call するだけ
    - これにより forward() 呼び出しごとに合成処理を繰り返す必要がなくなる
    """

    def __init__(
        self,
        base_model: nn.Module,
        list_of_task_vectors: List[FastTaskVector],
        lambdas: torch.Tensor
    ):
        super().__init__()
        self.device = torch.device("cuda")
        self.base_model = base_model
        self.task_vectors = list_of_task_vectors
        self.lambdas = lambdas.to(torch.float16).to(self.device)  # requires_grad=True
        

        # もしまだ base_model がGPU/FP16でない場合はここで変換
        for param in self.base_model.parameters():
            param.to(self.device).half()
            param.requires_grad = False  # 凍結

        # FastTaskVectorのdeltaもGPU/FP16化（必要なら）
        for tv in self.task_vectors:
            for k in tv.delta_dict:
                tv.delta_dict[k] = tv.delta_dict[k].to(self.device).half()

        # この辞書に "合成後パラメータ" をキャッシュする
        self.merged_params = None

    def merge(self):
        """
        現在のλ (self.lambdas) を用いて
        base_param + Σ_i (lambda_i * delta_i)
        を一回だけ計算し、self.merged_params に保持する。
        """
        merged_params = {}
        for (name, base_param) in self.base_model.named_parameters():
            # base_paramを明示的にGPUに移動
            base_param = base_param.to(self.device)
            
            total_delta = None
            for i, tv in enumerate(self.task_vectors):
                if name in tv.delta_dict:
                    # deltaも明示的にGPUに移動
                    delta_i = tv.delta_dict[name].to(self.device)
                    # lambda_i * delta_i は計算グラフ上で λ の勾配が生まれる
                    scaled = self.lambdas[i] * delta_i
                    if total_delta is None:
                        total_delta = scaled
                    else:
                        total_delta = total_delta + scaled

            if total_delta is None:
                # 差分がなければbase_paramのまま
                merged_params[name] = base_param
            else:
                merged_params[name] = base_param + total_delta

        # 結果をクラス変数にキャッシュ
        self.merged_params = merged_params

    def forward(self, input_ids, labels=None, **kwargs):
        """
        merge() で作った self.merged_params を使い、
        stateless.functional_call で forward を実行
        """
        if self.merged_params is None:
            raise RuntimeError("Please call .merge() before forward()")

        with autocast(enabled=True, dtype=torch.float16):
            outputs = stateless.functional_call(
                self.base_model,
                self.merged_params,
                args=(input_ids,),
                kwargs={"labels": labels, **kwargs}
            )

        # huggingface系モデルを想定 → loss があればそれを返す
        if hasattr(outputs, "loss"):
            return outputs.loss
        return outputs

###############################################################################
# メインの LambdaOptimizerCrossEntropy クラス (学習処理)
###############################################################################
class LambdaOptimizerCrossEntropy:
    """
    - モデルパラメータは凍結 (requires_grad=False)
    - λ (self.lambdas) のみ学習 (requires_grad=True)
    - Forward 時、LayerwiseMergedModelWrapper がベースパラメータに差分を加算
      => 計算グラフに (λ * delta) が乗るので、λ に勾配が伝播可能
    """

    def __init__(
        self,
        seed: int,
        pretrained_model,  # CPU上にあるベースモデル
        pretrained_model_name: str,
        tokenizers,
        models_to_merge: List,  # CPU上のファインチューニング済みモデルリスト
        initial_lambdas: Optional[np.ndarray] = None,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 1,
        num_train_samples: int = 4,
        optimizer_type: str = "adam",
        scheduler_type: str = "cosine",
        warmup_steps: int = 0,
        logger: Optional[logging.Logger] = None,
        params_name: str = None,
        finetuned_model_names: List[str] = None,
        optimized_lambda_filepath: str = None,
        max_length: int = 1024,
        **kwargs
    ):
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda")

        # 1) ベースモデルをFP16でGPUに読み込み
        self.pretrained_model = pretrained_model.to(torch.float16)
        # gradient checkpointingを有効化
        self.pretrained_model.gradient_checkpointing_enable()
        
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

        # 2) finetuned - pretrained の差分をCPU上で保持
        print("Calculating task vectors on GPU...")
        self.list_of_task_vectors = []
        with torch.no_grad():
            for i, ft_model in enumerate(models_to_merge, start=1):
                ft_model = ft_model.to(torch.float16)
                tv = FastTaskVector(self.pretrained_model, ft_model)
                self.list_of_task_vectors.append(tv)

        # 3) λ (タスク数) 初期化
        num_tasks = len(self.list_of_task_vectors)
        if initial_lambdas is None:
            initial_lambdas = np.ones(num_tasks, dtype=np.float16)  # 例: 全部1
        self.lambdas = torch.tensor(
            initial_lambdas,
            requires_grad=True,
            dtype=torch.float16,
            device=self.device
        )

        # 4) Optimizer & Scheduler
        self.setup_optimizer_and_scheduler(
            optimizer_type=optimizer_type,
            learning_rate=learning_rate,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            num_epochs=num_epochs
        )

        # 5) 訓練データ (GSM8K, MBPP, ja-MGSM) の読み込み
        print("Loading training data...")
        self.load_training_data(num_train_samples)

        # 6) その他設定
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizers = tokenizers
        self.finetuned_model_names = finetuned_model_names
        self.optimized_lambda_filepath = optimized_lambda_filepath

        # wandb (任意)
        self.setup_wandb(params_name)

        # 7) LayerwiseMergedModelWrapper
        self.merged_model = LayerwiseMergedModelWrapper(
            base_model=self.pretrained_model,
            list_of_task_vectors=self.list_of_task_vectors,
            lambdas=self.lambdas
        ).to(self.device)

        print("Initialization complete")

    def setup_wandb(self, params_name: str):
        """wandbの設定"""
        wandb.init(
            project="model_merging",
            name=f"lambda_optimization_{params_name}",
            config={
                "learning_rate": self.optimizer.param_groups[0]['lr'] if self.optimizer else None,
                "num_epochs": self.num_epochs,
                "batch_size": self.batch_size,
                "num_models": len(self.list_of_task_vectors),
                "model_names": self.finetuned_model_names
            }
        )
        self.wandb_run = wandb.run

    def setup_optimizer_and_scheduler(self, optimizer_type, learning_rate, scheduler_type, warmup_steps, num_epochs):
        """最適化関連の設定"""
        if optimizer_type == "adam":
            self.optimizer = torch.optim.Adam([self.lambdas], lr=learning_rate)
        elif optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD([self.lambdas], lr=learning_rate)
        else:
            self.optimizer = None  # spsa等、独自実装の場合は後で

        if scheduler_type == "cosine" and self.optimizer is not None:
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_epochs
            )
        else:
            self.scheduler = None

    def load_training_data(self, num_samples):
        """訓練データの読み込み"""
        self.gsm8k_data = load_gsm8k_data(num_samples=num_samples, seed=self.seed)
        self.mbpp_data  = load_mbpp_data(num_samples=num_samples, seed=self.seed)
        self.ja_mgsm_data = load_ja_mgsm_data(num_samples=num_samples, seed=self.seed)

    ###########################################################################
    # メモリモニタ用デコレータ (任意)
    ###########################################################################
    def memory_monitor_decorator(func):
        def wrapper(self, *args, **kwargs):
            if not torch.cuda.is_available():
                return func(self, *args, **kwargs)

            print(f"\n{'='*50}")
            print(f"Monitoring memory for: {func.__name__}")
            print(f"{'='*50}")

            # 実行前のメモリ状態
            print("\nBefore function execution:")
            for i in range(torch.cuda.device_count()):
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                free = total - allocated
                print(f"GPU {i}:")
                print(f"  Total:     {total:.2f} GB")
                print(f"  Allocated: {allocated:.2f} GB")
                print(f"  Reserved:  {reserved:.2f} GB")
                print(f"  Free:      {free:.2f} GB")

            # λの勾配状態（もし存在すれば）
            if hasattr(self, 'lambdas'):
                print(f"\nλ requires_grad: {self.lambdas.requires_grad}")
                print(f"Current λ values: {self.lambdas.detach().cpu().numpy()}")

            try:
                result = func(self, *args, **kwargs)

                # 実行後のメモリ状態
                print("\nAfter function execution:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"GPU {i}:")
                    print(f"  Allocated: {allocated:.2f} GB")
                    print(f"  Reserved:  {reserved:.2f} GB")

                # λの勾配（もし存在すれば）
                if hasattr(self, 'lambdas') and self.lambdas.grad is not None:
                    print(f"\nλ gradients: {self.lambdas.grad}")

                return result

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM Error in {func.__name__}!")
                    print("Memory state at error:")
                    for i in range(torch.cuda.device_count()):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        print(f"GPU {i}:")
                        print(f"  Allocated: {allocated:.2f} GB")
                        print(f"  Reserved:  {reserved:.2f} GB")
                raise e

        return wrapper

    ###########################################################################
    # GSM8K/MBPP/ja-MGSMの compute_*_batch_loss
    ###########################################################################
    #@memory_monitor_decorator
    def compute_gsm8k_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model, max_length=256) -> torch.Tensor:
        try:
            device = next(merged_model.parameters()).device
            
            # トークナイザーにパディングトークンが設定されていない場合、EOSトークンをパディングトークンとして使用
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            prompts, answers = [], []
            for item in batch_data:
                p = f"Q: {item['question']}\nA:"
                a = item["answer"]
                prompts.append(p)
                answers.append(a)

            prompt_enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            answer_enc = tokenizer(
                answers,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            pad_id = tokenizer.pad_token_id or 0

            input_ids_list, labels_list = [], []
            max_len = 0

            for i in range(len(prompts)):
                p_ids = prompt_enc.input_ids[i].tolist()
                a_ids = answer_enc.input_ids[i].tolist()

                p_len = sum(1 for x in p_ids if x != pad_id)
                a_len = sum(1 for x in a_ids if x != pad_id)

                merged_ids    = p_ids[:p_len] + a_ids[:a_len]
                merged_labels = ([-100]*p_len) + a_ids[:a_len]

                max_len = max(max_len, len(merged_ids))
                input_ids_list.append(merged_ids)
                labels_list.append(merged_labels)

            # パディング
            for i in range(len(input_ids_list)):
                diff_len = max_len - len(input_ids_list[i])
                input_ids_list[i].extend([pad_id]*diff_len)
                labels_list[i].extend([-100]*diff_len)

            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
            labels_tensor    = torch.tensor(labels_list,   dtype=torch.long, device=self.device)

            with autocast(enabled=True, dtype=torch.float16):
                outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
                # 計算グラフを保持したままlossをdetachせずにコピー
                loss = outputs.clone()

            # 勾配に関係ないテンソルを明示的に解放
            del input_ids_tensor, labels_tensor
            del prompt_enc, answer_enc
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

            return loss

        finally:
            # 例外発生時にもメモリを解放
            torch.cuda.empty_cache()

    #@memory_monitor_decorator
    def compute_mbpp_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model, max_length=256) -> torch.Tensor:
        try:
            device = next(merged_model.parameters()).device
            
            # トークナイザーにパディングトークンが設定されていない場合、EOSトークンをパディングトークンとして使用
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            

            prompts, codes = [], []
            for item in batch_data:
                prompts.append(f"{item['text']}\nSolution:\n")
                codes.append(item["code"])

            prompt_enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            code_enc   = tokenizer(
                codes,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            pad_id = tokenizer.pad_token_id or 0

            input_ids_list, labels_list = [], []
            max_len = 0

            for i in range(len(prompts)):
                p_ids = prompt_enc.input_ids[i].tolist()
                c_ids = code_enc.input_ids[i].tolist()

                p_len = sum(1 for x in p_ids if x != pad_id)
                c_len = sum(1 for x in c_ids if x != pad_id)

                merged_ids    = p_ids[:p_len] + c_ids[:c_len]
                merged_labels = ([-100]*p_len) + c_ids[:c_len]

                max_len = max(max_len, len(merged_ids))
                input_ids_list.append(merged_ids)
                labels_list.append(merged_labels)

            # パディング
            for i in range(len(input_ids_list)):
                diff_len = max_len - len(input_ids_list[i])
                input_ids_list[i].extend([pad_id]*diff_len)
                labels_list[i].extend([-100]*diff_len)

            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
            labels_tensor    = torch.tensor(labels_list,   dtype=torch.long, device=self.device)

            with autocast(enabled=True, dtype=torch.float16):
                outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
                # 計算グラフを保持したままlossをコピー
                loss = outputs.clone()

            # 勾配に関係ないテンソルのみを解放
            del input_ids_tensor, labels_tensor
            del prompt_enc, code_enc
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

            return loss

        finally:
            torch.cuda.empty_cache()

    #@memory_monitor_decorator
    def compute_ja_mgsm_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model, max_length=256) -> torch.Tensor:
        try:
            device = next(merged_model.parameters()).device
            
            # トークナイザーにパディングトークンが設定されていない場合、EOSトークンをパディングトークンとして使用
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            

            prompts, answers = [], []
            for item in batch_data:
                prompts.append(f"{item['question']}\n答えを考える: ")
                answers.append(item["answer"])

            prompt_enc = tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            answer_enc = tokenizer(
                answers,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)

            pad_id = tokenizer.pad_token_id or 0

            input_ids_list, labels_list = [], []
            max_len = 0

            for i in range(len(prompts)):
                p_ids = prompt_enc.input_ids[i].tolist()
                a_ids = answer_enc.input_ids[i].tolist()

                p_len = sum(1 for x in p_ids if x != pad_id)
                a_len = sum(1 for x in a_ids if x != pad_id)

                merged_ids    = p_ids[:p_len] + a_ids[:a_len]
                merged_labels = ([-100]*p_len) + a_ids[:a_len]

                max_len = max(max_len, len(merged_ids))
                input_ids_list.append(merged_ids)
                labels_list.append(merged_labels)

            # パディング
            for i in range(len(input_ids_list)):
                diff_len = max_len - len(input_ids_list[i])
                input_ids_list[i].extend([pad_id]*diff_len)
                labels_list[i].extend([-100]*diff_len)

            input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
            labels_tensor    = torch.tensor(labels_list,   dtype=torch.long, device=self.device)

            with autocast(enabled=True, dtype=torch.float16):
                outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
                # 計算グラフを保持したままlossをコピー
                loss = outputs.clone()

            # 勾配に関係ないテンソルのみを解放
            del input_ids_tensor, labels_tensor
            del prompt_enc, answer_enc
            del outputs
            gc.collect()
            torch.cuda.empty_cache()

            return loss

        finally:
            torch.cuda.empty_cache()

    ###########################################################################
    # エポック学習処理 (train_one_epoch)
    ###########################################################################
    def make_batches(self, data_list, bsz):
        return [data_list[i:i+bsz] for i in range(0, len(data_list), bsz)]

    #@memory_monitor_decorator
    def train_one_epoch(self, batch_size: int = 2):
        print("\n=== train_one_epoch start ===")
        print(f"λの勾配状態: {self.lambdas.requires_grad}")
        
        self.merged_model.merge()
                
        # optimizer.zero_gradをループの外に出す
        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float16, requires_grad=True)

        # タスクごとにバッチを回す
        tasks = [
            ("gsm8k",   self.gsm8k_data,   self.tokenizers[0], self.compute_gsm8k_batch_loss),
            ("mbpp",    self.mbpp_data,    self.tokenizers[1], self.compute_mbpp_batch_loss),
            ("ja_mgsm", self.ja_mgsm_data, self.tokenizers[2], self.compute_ja_mgsm_batch_loss)
        ]

        try:            
            for task_name, data, tokenizer, compute_loss_fn in tasks:
                print(f"\nProcessing task: {task_name}")
                batches = self.make_batches(data, batch_size)
                task_loss = torch.tensor(0.0, device=self.device, dtype=torch.float16, requires_grad=True)

                for batch_idx, batch in enumerate(batches):
                    print(f"\nBatch {batch_idx + 1}/{len(batches)}")
                    batch_loss = compute_loss_fn(batch, tokenizer, self.merged_model)
                    print(f"Batch loss: {batch_loss.item()}")
                    print(f"Batch loss requires_grad: {batch_loss.requires_grad}")
                    task_loss = task_loss + batch_loss
                    
                    del batch
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                print(f"Task {task_name} total loss: {task_loss.item()}")
                total_loss = total_loss + task_loss

            # backward()実行
            total_loss.backward()
            
            # optimizerのstepを実行
            self.optimizer.step()
            
            # schedulerがあれば更新
            if self.scheduler is not None:
                self.scheduler.step()
            
            gc.collect()
            torch.cuda.empty_cache()

        finally:
            torch.cuda.empty_cache()

        return total_loss.item()

    ###########################################################################
    # 学習全体 (optimize)
    ###########################################################################
    #@memory_monitor_decorator
    def optimize(self):
        best_loss = float("inf")
        best_lambdas = None

        for epoch in range(self.num_epochs):
            epoch_loss = self.train_one_epoch(batch_size=self.batch_size)

            self.wandb_run.log({
                "epoch": epoch,
                "loss": epoch_loss,
                **{f"lambda_{i}": float(self.lambdas[i].item()) for i in range(len(self.lambdas))}
            })

            # CSV保存
            if self.optimized_lambda_filepath:
                optimized_df = pd.DataFrame({
                    'epoch': [epoch + 1],
                    'model_name': [self.finetuned_model_names],
                    'lambdas': [self.lambdas.detach().cpu().numpy().tolist()],
                    'loss': [epoch_loss]
                })
                if os.path.exists(self.optimized_lambda_filepath):
                    optimized_df.to_csv(self.optimized_lambda_filepath, mode='a', header=False, index=False)
                else:
                    optimized_df.to_csv(self.optimized_lambda_filepath, index=False)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_lambdas = self.lambdas.clone()

            print(f"[Epoch {epoch+1}/{self.num_epochs}] loss={epoch_loss:.4f}, "
                  f"lambdas={[float(l.item()) for l in self.lambdas]}")

        # ベスト結果
        if best_lambdas is not None:
            final_lams = best_lambdas.detach().cpu().numpy()
        else:
            final_lams = self.lambdas.detach().cpu().numpy()

        self.wandb_run.log({
            "best_loss": best_loss,
            **{f"best_lambda_{i}": val for i, val in enumerate(final_lams)}
        })

        return final_lams

    ###########################################################################
    # (任意) Optuna の実装例
    ###########################################################################
    #@memory_monitor_decorator
    def optimize_with_optuna(self, n_trials: int = 100) -> np.ndarray:
        def objective(trial):
            # λのサンプリング（正規化なし）
            lambdas = []
            for i in range(len(self.list_of_task_vectors)):
                val = trial.suggest_float(f'lambda_{i}', 0.0, 1.0)
                lambdas.append(val)

            # モデルに反映（requires_grad=Falseで設定）
            self.lambdas.data = torch.tensor(lambdas, dtype=torch.float16, device=self.device)
            self.lambdas.requires_grad = False  # 勾配計算を無効化
            self.merged_model.merge()
            self.merged_model.eval()

            with torch.no_grad():
                total_loss = torch.tensor(0.0, device=self.device, dtype=torch.float16)
                
                tasks = [
                    ("gsm8k",   self.gsm8k_data,   self.tokenizers[0], self.compute_gsm8k_batch_loss),
                    ("mbpp",    self.mbpp_data,    self.tokenizers[1], self.compute_mbpp_batch_loss),
                    ("ja_mgsm", self.ja_mgsm_data, self.tokenizers[2], self.compute_ja_mgsm_batch_loss)
                ]

                for task_name, data, tokenizer, compute_loss_fn in tasks:
                    batches = self.make_batches(data, self.batch_size)
                    task_loss = torch.tensor(0.0, device=self.device, dtype=torch.float16)

                    for batch in batches:
                        batch_loss = compute_loss_fn(batch, tokenizer, self.merged_model)
                        task_loss = task_loss + batch_loss

                    total_loss = total_loss + task_loss

                # メモリ解放
                torch.cuda.empty_cache()
                
                return total_loss.item()

        # Optunaの設定
        study = optuna.create_study(
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        
        # 探索の実行
        study.optimize(
            objective, 
            n_trials=n_trials,
            callbacks=[
                lambda study, trial: wandb.log({
                    "best_loss": study.best_value,
                    **{f"best_lambda_{i}": v for i, v in enumerate(study.best_params.values())}
                }) if self.wandb_run else None
            ]
        )

        # 最適なλを取得
        best_lambdas = [study.best_params[f'lambda_{i}'] for i in range(len(self.list_of_task_vectors))]
        best_lambdas = np.array(best_lambdas)
        best_loss = study.best_value

        # 結果をログ出力
        print(f"\n[Optuna] Best Trial: {study.best_trial.number}")
        print(f"[Optuna] Best Loss: {best_loss:.4f}")
        print(f"[Optuna] Best Lambdas: {best_lambdas}")

        # CSVに保存
        if self.optimized_lambda_filepath:
            optimized_df = pd.DataFrame({
                'epoch': [1],  # Optunaでは1エポックとして扱う
                'model_name': [self.finetuned_model_names],
                'lambdas': [best_lambdas.tolist()],
                'loss': [best_loss]
            })
            if os.path.exists(self.optimized_lambda_filepath):
                optimized_df.to_csv(self.optimized_lambda_filepath, mode='a', header=False, index=False)
            else:
                optimized_df.to_csv(self.optimized_lambda_filepath, index=False)

        # 最適なλをモデルに�定
        self.lambdas.data = torch.tensor(best_lambdas, dtype=torch.float16, device=self.device)
        self.lambdas.requires_grad = False
        self.merged_model.merge()

        return best_lambdas


