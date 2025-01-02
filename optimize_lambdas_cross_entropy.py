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

# 重要: stateless.functional_call を使う
import torch.nn.utils.stateless as stateless

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
# 以下はサンプルの読み込み関数
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
# ベースモデルとファインチューニング済みモデルの差分を取るクラス(タスクベクトル)
###############################################################################
class NewTaskVector:
    """
    pretrained_model と finetuned_model の差分をとった辞書: 
      'param_name' -> GPU上に保持される Tensor(差分)
    """
    def __init__(self, pretrained_model, finetuned_model, exclude_param_names_regex=[]):
        self.task_vector_param_dict = {}
        with torch.no_grad():
            is_4bit_base = hasattr(pretrained_model, 'is_loaded_in_4bit') and pretrained_model.is_loaded_in_4bit
            is_4bit_ft   = hasattr(finetuned_model, 'is_loaded_in_4bit') and finetuned_model.is_loaded_in_4bit

            for (n1, p1), (n2, p2) in zip(pretrained_model.named_parameters(), finetuned_model.named_parameters()):
                # 必要に応じてfloat16 GPU上へ転送
                if is_4bit_base:
                    p1_data = p1.to(dtype=torch.float16)
                else:
                    p1_data = p1.to(device='cuda', dtype=torch.float16)

                if is_4bit_ft:
                    p2_data = p2.to(dtype=torch.float16)
                else:
                    p2_data = p2.to(device='cuda', dtype=torch.float16)

                # 差分をGPU上に保持
                diff = p2_data - p1_data
                self.task_vector_param_dict[n1] = diff  # GPU上

###############################################################################
# forward 時に (base_param + Σ(λ_i * diff_i)) を合成するラッパークラス
###############################################################################
class MergedModelWrapper(nn.Module):
    """
    base_param + Σ(λ_i * diff_i) を一度にGPU上で合成してforwardするラッパー
    """
    def __init__(self, pretrained_model, task_vectors, lambdas):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.task_vectors = task_vectors
        self.lambdas = lambdas  # 学習対象 (requires_grad=True) を想定

        self.is_4bit = hasattr(pretrained_model, 'is_loaded_in_4bit') and pretrained_model.is_loaded_in_4bit

        # ベースモデルをfreeze
        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, labels=None, **kwargs):
        """
        1. ベースパラメータをGPUへコピー (4bitの場合はfloat16変換)
        2. 全タスクベクトル(diff)を GPU 上で足し合わせる
        3. functional_callでforward
        """
        device = self.lambdas.device
        input_ids = input_ids.to(device)
        if labels is not None:
            labels = labels.to(device)

        # 合成後パラメータを保持する辞書
        param_dict = {}

        with torch.set_grad_enabled(self.training), autocast(enabled=True):
            for name, base_param in self.pretrained_model.named_parameters():
                # base_paramをGPUへ(float16)
                if self.is_4bit:
                    base_gpu = base_param.to(dtype=torch.float16)
                else:
                    base_gpu = base_param.to(device=device, dtype=torch.float16)

                base_gpu = base_gpu.clone()

                # 各task_vector(diff)を直接加算（すでにGPU上にある）
                for i, tv in enumerate(self.task_vectors):
                    diff = tv.task_vector_param_dict[name]  # GPU上
                    lam = self.lambdas[i].to(dtype=torch.float16)
                    base_gpu += lam * diff

                param_dict[name] = base_gpu

        # stateless.functional_callでforward
        outputs = stateless.functional_call(
            self.pretrained_model.to(device),
            param_dict,
            (input_ids,),
            {'labels': labels, **kwargs}
        )
        return outputs


###############################################################################
# メインの LambdaOptimizer クラス (プロンプト部分を -100 ラベルにする)
###############################################################################
class LambdaOptimizerCrossEntropy:
    """
    - モデルパラメータは凍結 (requires_grad=False)
    - λ (self.lambdas) のみ学習 (requires_grad=True)
    - 合成: base_param + sum(lambda_i * diff_i)
    - プロンプト部分を -100, 回答部分を実ラベルにする
    """

    def __init__(
        self,
        seed: int,
        pretrained_model,       # transformers の PreTrainedModel (4bitモデルの可能性あり)
        pretrained_model_name: str,
        tokenizers,             # [tok_gsm8k, tok_mbpp, tok_ja_mgsm]
        models_to_merge: List,  # finetuned models (4bitの可能性あり)
        initial_lambdas: Optional[np.ndarray] = None,
        num_epochs: int = 10,
        learning_rate: float = 0.001,
        batch_size: int = 2,
        num_train_samples: int = 4,
        optimizer_type: str = "adam",
        scheduler_type: str = "cosine",
        warmup_steps: int = 0,
        logger: Optional[logging.Logger] = None,
        params_name: str = None,
        finetuned_model_names: List[str] = None,
        optimized_lambda_filepath: str = None,
        max_length: int = 256,
    ):
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # デバッグ用にmodels_to_mergeに同じモデルが3つ入っている場合、それぞれのパラメータを少し変更する
        for i, model in enumerate(models_to_merge):
            for name, param in model.named_parameters():
                with torch.no_grad():
                    param.add_(torch.randn_like(param) * 0.1 * (i + 1))

        self.pretrained_model = pretrained_model
        # 4bitモデルでない場合は .to(device)
        if not (hasattr(self.pretrained_model, 'is_loaded_in_4bit') and self.pretrained_model.is_loaded_in_4bit):
            self.pretrained_model = self.pretrained_model.to(self.device)

        self.models_to_merge = []
        for model in models_to_merge:
            if not (hasattr(model, 'is_loaded_in_4bit') and model.is_loaded_in_4bit):
                model = model.to(self.device)
            self.models_to_merge.append(model)

        self.pretrained_model_name = pretrained_model_name
        self.tokenizers = tokenizers  # [tok_gsm8k, tok_mbpp, tok_ja_mgsm]
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_train_samples = num_train_samples
        self.max_length = max_length

        # λ の初期値
        if initial_lambdas is None:
            initial_lambdas = np.zeros(len(models_to_merge))

        # Task Vectorsの初期化
        self.task_vectors = []
        with torch.no_grad():
            for ft_model in models_to_merge:
                tv = NewTaskVector(self.pretrained_model, ft_model)
                self.task_vectors.append(tv)

        # λ (GPU上に配置)
        self.lambdas = torch.tensor(
            initial_lambdas, 
            requires_grad=True, 
            dtype=torch.float16, 
            device=self.device
        )

        # 事前学習モデルのパラメータを凍結
        for p in pretrained_model.parameters():
            p.requires_grad = False

        # Optimizerの設定
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam([self.lambdas], lr=self.learning_rate)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD([self.lambdas], lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # スケジューラーの設定
        self.scheduler_type = scheduler_type
        total_steps = num_epochs * (num_train_samples // batch_size + 1) * 3  # 3タスク程度

        if scheduler_type == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6
            )
        elif scheduler_type == "cosine_warmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None

        # シード
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # データ (各10件程度をサンプリング)
        self.gsm8k_data = load_gsm8k_data(num_samples=self.num_train_samples, seed=seed)
        self.mbpp_data = load_mbpp_data(num_samples=self.num_train_samples, seed=seed)
        self.ja_mgsm_data = load_ja_mgsm_data(num_samples=self.num_train_samples, seed=seed)

        # wandb 初期化
        self.wandb_run = wandb.init(
            project="model_merging",
            name=f"LambdaOptimizerCrossEntropy_{seed}",
            config={
                "seed": seed,
                "pretrained_model": pretrained_model_name,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "num_train_samples": num_train_samples,
                "optimizer_type": optimizer_type,
                "scheduler_type": scheduler_type,
                "warmup_steps": warmup_steps
            }
        )

        # 追加のパラメータ
        self.params_name = params_name
        self.finetuned_model_names = finetuned_model_names
        self.optimized_lambda_filepath = optimized_lambda_filepath

    def memory_tracker(func):
        """関数のメモリ使用量を追跡するデコレータ"""
        def wrapper(self, *args, **kwargs):
            if not torch.cuda.is_available():
                return func(self, *args, **kwargs)
            
            torch.cuda.empty_cache()
            start_allocated = torch.cuda.memory_allocated() / 1024**3
            start_reserved  = torch.cuda.memory_reserved()  / 1024**3
            
            result = func(self, *args, **kwargs)
            
            end_allocated = torch.cuda.memory_allocated() / 1024**3
            end_reserved  = torch.cuda.memory_reserved()  / 1024**3
            
            print(f"\n[Memory Usage] {func.__name__}")
            print(f"  Before - Allocated: {start_allocated:.2f}GB, Reserved: {start_reserved:.2f}GB")
            print(f"  After  - Allocated: {end_allocated:.2f}GB, Reserved: {end_reserved:.2f}GB")
            print(f"  Diff   - Allocated: {end_allocated - start_allocated:.2f}GB, Reserved: {end_reserved - start_reserved:.2f}GB")
            
            return result
        return wrapper

    def debug_decorator(func):
        """関数の実行状況とメモリ使用量を表示するデコレータ"""
        def wrapper(self, *args, **kwargs):
            print(f"\n{'='*80}")
            print(f"Executing: {func.__name__}")
            print(f"{'='*80}")
            
            # GPU メモリ
            if torch.cuda.is_available():
                gpu_start_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_start_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"GPU Memory before {func.__name__}:")
                print(f"  Allocated: {gpu_start_allocated:.2f} GB")
                print(f"  Reserved:  {gpu_start_reserved:.2f} GB")
            
            # RAM
            process = psutil.Process(os.getpid())
            ram_start = process.memory_info().rss / 1024**3
            print(f"RAM Usage before {func.__name__}: {ram_start:.2f} GB")
            
            # 関数実行
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            
            print(f"\nFinished: {func.__name__}")
            print(f"Execution time: {end_time - start_time:.2f} seconds")
            
            # GPU メモリ
            if torch.cuda.is_available():
                gpu_end_allocated = torch.cuda.memory_allocated() / 1024**3
                gpu_end_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"\nGPU Memory after {func.__name__}:")
                print(f"  Allocated: {gpu_end_allocated:.2f} GB")
                print(f"  Reserved:  {gpu_end_reserved:.2f} GB")
                print(f"  Diff Allocated: {gpu_end_allocated - gpu_start_allocated:.2f} GB")
                print(f"  Diff Reserved:  {gpu_end_reserved - gpu_start_reserved:.2f} GB")
            
            # RAM
            ram_end = process.memory_info().rss / 1024**3
            print(f"RAM Usage after {func.__name__}: {ram_end:.2f} GB")
            print(f"RAM Diff: {ram_end - ram_start:.2f} GB")
            print(f"{'='*80}\n")
            
            return result
        return wrapper

    ############################################################################
    # GSM8K/MBPP/ja-MGSMの compute_*_batch_loss
    ############################################################################
    @debug_decorator
    def compute_gsm8k_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model, max_length=256) -> torch.Tensor:
        device = next(merged_model.parameters()).device

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

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(device)
        labels_tensor    = torch.tensor(labels_list,   dtype=torch.long).to(device)

        with torch.set_grad_enabled(True):
            outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
            loss = outputs.loss
            
            if not loss.requires_grad:
                print("Warning: Loss requires_grad is False!")
                loss = loss.detach().requires_grad_(True)
            
            return loss

    @debug_decorator
    def compute_mbpp_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model, max_length=256) -> torch.Tensor:
        device = next(merged_model.parameters()).device

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

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(device)
        labels_tensor    = torch.tensor(labels_list,   dtype=torch.long).to(device)

        with torch.set_grad_enabled(True):
            outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
            loss = outputs.loss
            
            if not loss.requires_grad:
                print("Warning: Loss requires_grad is False!")
                loss = loss.detach().requires_grad_(True)
            
            return loss

    @debug_decorator
    def compute_ja_mgsm_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model, max_length=256) -> torch.Tensor:
        device = next(merged_model.parameters()).device

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

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long).to(device)
        labels_tensor    = torch.tensor(labels_list,   dtype=torch.long).to(device)

        with torch.set_grad_enabled(True):
            outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
            loss = outputs.loss
            
            if not loss.requires_grad:
                print("Warning: Loss requires_grad is False!")
                loss = loss.detach().requires_grad_(True)
            
            return loss

    ###########################################################################
    # エポック学習処理 (train_one_epoch)
    ###########################################################################
    def print_gpu_memory(self, message=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved  = torch.cuda.memory_reserved()  / 1024**3
            print(f"GPU Memory [{message}]")
            print(f"  Allocated: {allocated:.2f}GB")
            print(f"  Reserved:  {reserved:.2f}GB")

    def make_batches(self, data_list, bsz):
        return [data_list[i:i+bsz] for i in range(0, len(data_list), bsz)]

    @debug_decorator
    def train_one_epoch(self, batch_size: int = 2):
        print("\n=== train_one_epoch start ===")
        print(f"Initial λ values: {self.lambdas.detach().cpu().numpy()}")

        self.optimizer.zero_grad()

        # MergedModelWrapper を生成 (すべてGPU上で実行)
        merged_model = MergedModelWrapper(
            pretrained_model=(
                self.pretrained_model 
                if (hasattr(self.pretrained_model, 'is_loaded_in_4bit') and self.pretrained_model.is_loaded_in_4bit) 
                else self.pretrained_model.to(self.device)
            ),
            task_vectors=self.task_vectors,
            lambdas=self.lambdas
        )
        merged_model.to(self.device)

        tasks = [
            ("gsm8k",   self.gsm8k_data,   self.tokenizers[0], self.compute_gsm8k_batch_loss),
            ("mbpp",    self.mbpp_data,    self.tokenizers[1], self.compute_mbpp_batch_loss),
            ("ja_mgsm", self.ja_mgsm_data, self.tokenizers[2], self.compute_ja_mgsm_batch_loss),
        ]

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task_name, data, tokenizer, compute_loss_fn in tasks:
            batches = self.make_batches(data, batch_size)
            task_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

            for i, batch in enumerate(batches):
                batch_loss = compute_loss_fn(batch, tokenizer, merged_model, self.max_length)
                task_loss = task_loss + batch_loss

            task_loss = task_loss / len(batches)
            total_loss = total_loss + task_loss

            print(f"[{task_name}] loss = {task_loss.item():.4f}")

        total_loss.backward()
        print(f"Total loss = {total_loss.item():.4f}")

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        print(f"Updated λ values: {self.lambdas.detach().cpu().numpy()}")
        return total_loss.item()

    ###########################################################################
    # 通常の学習
    ###########################################################################
    @debug_decorator
    def optimize(self):
        best_loss = float("inf")
        best_lambdas = None

        # (任意) MixedPrecision: 学習全体をスケーリングするなら GradScaler を作成
        scaler = GradScaler()

        for epoch in range(self.num_epochs):
            # ここでは train_one_epoch 内部で backward() しているため、
            # autocast + GradScalerパターンと干渉しない実装になっています。
            epoch_loss = self.train_one_epoch(batch_size=self.batch_size)

            self.wandb_run.log({
                "epoch": epoch,
                "loss": epoch_loss,
                **{f"lambda_{i}": self.lambdas[i].item() for i in range(len(self.lambdas))}
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
                  f"lambdas={[l.item() for l in self.lambdas]}")

        # ベスト結果
        if best_lambdas is not None:
            final_lams = best_lambdas.detach().cpu().numpy()
        else:
            final_lams = self.lambdas.detach().cpu().numpy()

        self.wandb_run.log({
            "best_loss": best_loss,
            **{f"best_lambda_{i}": val for i, val in enumerate(final_lams)}
        })

        self.wandb_run.finish()
        return final_lams

    ###########################################################################
    # (任意) Optuna の実装例
    ###########################################################################
    @debug_decorator
    def optimize_with_optuna(self, n_trials: int = 30) -> np.ndarray:
        def objective(trial):
            # lambda の探索
            lambdas = []
            for i in range(len(self.models_to_merge)):
                val = trial.suggest_float(f'lambda_{i}', 0.0, 1.0)
                lambdas.append(val)
            
            device = next(self.pretrained_model.parameters()).device
            self.lambdas = torch.tensor(lambdas, dtype=torch.float16, device=device, requires_grad=True)

            # 簡易的に1エポックだけ行って loss を返す
            epoch_loss = self.train_one_epoch(batch_size=self.batch_size)
            return epoch_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_lambdas = [study.best_params[f'lambda_{i}'] for i in range(len(self.models_to_merge))]
        best_loss = study.best_value

        # 反映
        device = next(self.pretrained_model.parameters()).device
        self.lambdas = torch.tensor(best_lambdas, dtype=torch.float16, device=device, requires_grad=True)

        self.wandb_run.log({
            "optuna_best_trial": study.best_trial.number,
            "optuna_best_loss": best_loss,
            **{f"optuna_best_lambda_{i}": v for i, v in enumerate(best_lambdas)}
        })

        if self.optimized_lambda_filepath:
            if not os.path.exists(self.optimized_lambda_filepath):
                pd.DataFrame(columns=["model_name","lambdas","loss"]).to_csv(self.optimized_lambda_filepath, index=False)
            optimized_df = pd.DataFrame({
                'model_name': [self.finetuned_model_names],
                'lambdas': [best_lambdas],
                'loss': [best_loss]
            })
            optimized_df.to_csv(self.optimized_lambda_filepath, mode='a', header=False, index=False)

        print(f"[Optuna] Best Trial: {study.best_trial.number}, Best Loss: {best_loss:.4f}")
        print(f"[Optuna] Best Lambdas: {best_lambdas}")

        return np.array(best_lambdas)
