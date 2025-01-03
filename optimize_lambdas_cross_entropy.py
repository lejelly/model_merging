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
# 1) 差分をCPU上に保持するクラス (変更なし)
###############################################################################
class CPUStoredTaskVector:
    """
    「finetuned_model - pretrained_model」の差分を CPU 上に保持。
    param_name -> CPU上 Tensor(差分)
    """
    def __init__(self, pretrained_model: nn.Module, finetuned_model: nn.Module):
        self.delta_dict = {}
        with torch.no_grad():
            for (name_pre, p_pre), (name_ft, p_ft) in zip(
                pretrained_model.named_parameters(),
                finetuned_model.named_parameters()
            ):
                # パラメータ名が一致しているか確認
                assert name_pre == name_ft, f"Param mismatch: {name_pre} vs {name_ft}"
                # CPUに移動して差分を float32 で保持（VRAM節約）
                delta = (p_ft.cpu().float() - p_pre.cpu().float())
                self.delta_dict[name_pre] = delta

###############################################################################
# 2) forward 時にレイヤーごとに合成するラッパークラス (修正ポイント)
###############################################################################
class LayerwiseMergedModelWrapper(nn.Module):
    """
    - base_model: GPU上 (frozen) のベースモデル
    - list_of_task_vectors: CPU上に保存された "param_name -> delta" の辞書リスト
    - lambdas: nn.Parameter or Tensor (学習対象: 各タスクの重み)

    以前のコードでは forward_pre_hook 内で `param.data.add_(...)` をしていましたが
    autograd が勾配を追跡できないため lambda に勾配が流れませんでした。

    下記の修正版では:
      - pre_hook で「param -> param + Σ(lambdas[i]*delta)」と演算し、モジュールの _parameters を一時差し替え
      - post_hook で元に戻す
    というフローで、計算グラフに乗せています。
    """

    def __init__(self, base_model: nn.Module, list_of_task_vectors: List[CPUStoredTaskVector], lambdas: torch.Tensor):
        super().__init__()
        self.base_model = base_model.to(torch.float16)
        # gradient checkpointingを有効化
        self.base_model.gradient_checkpointing_enable()
        
        for p in self.base_model.parameters():
            p.requires_grad = False

        self.list_of_task_vectors = list_of_task_vectors
        self.lambdas = lambdas.to(torch.float16)
        
        self._original_params = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """
        モジュールごとに forward_pre_hook / forward_hook を仕掛ける。
        forward_pre_hook で差分を加算 → forward_hook で差分を引いて戻す ではなく、
        - pre_hookでモジュールのparamを「merged_param (= param + Σ(lambdas*delta))」に差し替え
        - post_hookで元に戻す
        """
        for name, module in self.base_model.named_modules():
            # このモジュールにパラメータがあるならフックを付ける
            params_in_module = list(module.named_parameters(recurse=False))
            if len(params_in_module) > 0:
                pre_hook = module.register_forward_pre_hook(self._make_pre_hook(name))
                post_hook = module.register_forward_hook(self._make_post_hook(name))
                self.hooks.append(pre_hook)
                self.hooks.append(post_hook)

    def _make_pre_hook(self, module_prefix: str):
        """
        レイヤーの forward 前 に「param -> param + Σ(lambdas * delta)」を差し替え
        """
        def pre_hook(module, inputs):
            for (param_name, param) in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}"
                self._original_params[full_name] = param

                # 1. 一時変数をすぐに解放するcontextmanagerを使用
                with torch.cuda.amp.autocast():
                    merged_param = param.clone()
                    for i, tv in enumerate(self.list_of_task_vectors):
                        if full_name in tv.delta_dict:
                            # 2. デルタを1つずつ処理して即解放
                            delta = tv.delta_dict[full_name].to(
                                param.device,
                                dtype=torch.float16,
                                non_blocking=True
                            )
                            merged_param.add_(self.lambdas[i] * delta)
                            del delta

                # 3. 計算グラフに乗せる最後の演算のみrequires_grad=True
                module._parameters[param_name] = merged_param

        return pre_hook

    def _make_post_hook(self, module_prefix: str):
        """
        レイヤーの forward 後 に元のパラメータに戻す
        """
        def post_hook(module, inputs, output):
            for (param_name, param) in module.named_parameters(recurse=False):
                full_name = f"{module_prefix}.{param_name}"
                if full_name in self._original_params:
                    # 4. 元のパラメータを戻す際に一時変数を即解放
                    orig_param = self._original_params[full_name]
                    module._parameters[param_name] = orig_param
                    del self._original_params[full_name]
            return output
        return post_hook

    def forward(self, input_ids, labels=None, **kwargs):
        """
        これで (base_param + Σ(lambdas * delta)) を使った forward を行う。
        """
        # 5. forward実行後に不要な中間表現を解放
        with autocast(enabled=True, dtype=torch.float16):
            outputs = self.base_model(input_ids=input_ids, labels=labels, **kwargs)
            if hasattr(outputs, 'loss'):
                loss = outputs.loss
                # 必要な値以外は即解放
                del outputs.logits
                del outputs.past_key_values
                torch.cuda.empty_cache()
                return loss
            return outputs

    def cleanup(self):
        """フックを外す (不要ならそのままでもOK)"""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def __del__(self):
        self.cleanup()

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
        max_length: int = 256,
        **kwargs
    ):
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda")

        # 1) ベースモデルを GPU にロード (勾配不要)
        self.pretrained_model = pretrained_model.to(self.device)
        for p in self.pretrained_model.parameters():
            p.requires_grad = False

        # 2) finetuned - pretrained の差分をCPU上で保持
        print("Calculating task vectors (CPU deltas)...")
        self.list_of_task_vectors = []
        with torch.no_grad():
            for i, ft_model in enumerate(models_to_merge, start=1):
                print(f"Processing model {i}/{len(models_to_merge)} -> CPU delta")
                ft_model = ft_model.to(self.device)

                tv = CPUStoredTaskVector(self.pretrained_model, ft_model)
                self.list_of_task_vectors.append(tv)

                # finetuned_model 不要になったらメモリを開放
                ft_model.cpu()
                del ft_model
                torch.cuda.empty_cache()
                gc.collect()

        # 参照破棄
        models_to_merge.clear()
        del models_to_merge
        torch.cuda.empty_cache()
        gc.collect()

        # 3) λ (タスク数) 初期化
        num_tasks = len(self.list_of_task_vectors)
        if initial_lambdas is None:
            initial_lambdas = np.ones(num_tasks, dtype=np.float32)  # 例: 全部1
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
    @memory_monitor_decorator
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

        with autocast(enabled=True, dtype=torch.float16):
            outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
            loss = outputs
            if not loss.requires_grad:
                loss = loss.detach().requires_grad_(True)
            return loss

    @memory_monitor_decorator
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

        with autocast(enabled=True, dtype=torch.float16):
            outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
            loss = outputs
            if not loss.requires_grad:
                loss = loss.detach().requires_grad_(True)
            return loss

    @memory_monitor_decorator
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

        with autocast(enabled=True, dtype=torch.float16):
            outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
            loss = outputs
            if not loss.requires_grad:
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

    @memory_monitor_decorator
    def train_one_epoch(self, batch_size: int = 2):
        print("\n=== train_one_epoch start ===")
        print(f"λの勾配状態: {self.lambdas.requires_grad}")
        print(f"Initial λ values: {self.lambdas.detach().cpu().numpy()}")

        self.optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # タスクごとにバッチを回す
        tasks = [
            ("gsm8k",   self.gsm8k_data,   self.tokenizers[0], self.compute_gsm8k_batch_loss),
            ("mbpp",    self.mbpp_data,    self.tokenizers[1], self.compute_mbpp_batch_loss),
            ("ja_mgsm", self.ja_mgsm_data, self.tokenizers[2], self.compute_ja_mgsm_batch_loss)
        ]

        try:
            self.merged_model.to(self.device)

            for task_name, data, tokenizer, compute_loss_fn in tasks:
                batches = self.make_batches(data, batch_size)
                task_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                for batch in batches:
                    batch_loss = compute_loss_fn(batch, tokenizer, self.merged_model)
                    task_loss = task_loss + batch_loss

                total_loss = total_loss + task_loss

            # 勾配計算
            total_loss.backward()

            # λの勾配確認
            print(f"λの勾配値: {self.lambdas.grad}")

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        finally:
            self.merged_model.cpu()
            torch.cuda.empty_cache()

        return total_loss.item()

    ###########################################################################
    # 学習全体 (optimize)
    ###########################################################################
    @memory_monitor_decorator
    def optimize(self):
        best_loss = float("inf")
        best_lambdas = None

        # MixedPrecisionでまとめてやる場合 (本実装ではステップごとにautocast済みなので省略可)
        scaler = GradScaler()

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
    @memory_monitor_decorator
    def optimize_with_optuna(self, n_trials: int = 30) -> np.ndarray:
        def objective(trial):
            # 新しいλをサンプリング
            lambdas = []
            for i in range(len(self.list_of_task_vectors)):
                val = trial.suggest_float(f'lambda_{i}', 0.0, 1.0)
                lambdas.append(val)

            # 反映
            self.lambdas.data = torch.tensor(lambdas, dtype=torch.float32, device=self.device)

            # 1エポックだけ学習
            epoch_loss = self.train_one_epoch(batch_size=self.batch_size)
            return epoch_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_lambdas = [study.best_params[f'lambda_{i}'] for i in range(len(self.list_of_task_vectors))]
        best_loss = study.best_value

        # 反映
        self.lambdas.data = torch.tensor(best_lambdas, dtype=torch.float32, device=self.device)

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
