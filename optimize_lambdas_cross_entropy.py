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
      'param_name' -> Tensor(差分)
    """
    def __init__(self, pretrained_model, finetuned_model, exclude_param_names_regex=[]):
        self.task_vector_param_dict = {}
        with torch.no_grad():  # 勾配計算不要
            is_8bit_base = hasattr(pretrained_model, 'is_loaded_in_8bit') and pretrained_model.is_loaded_in_8bit
            is_8bit_ft   = hasattr(finetuned_model, 'is_loaded_in_8bit') and finetuned_model.is_loaded_in_8bit

            for (n1, p1), (n2, p2) in zip(pretrained_model.named_parameters(), finetuned_model.named_parameters()):
                # (8-bit) Convert to float on CPU if needed
                if is_8bit_base:
                    p1_data = p1.float().cpu()
                else:
                    p1_data = p1.cpu().float()

                if is_8bit_ft:
                    p2_data = p2.float().cpu()
                else:
                    p2_data = p2.cpu().float()

                diff = (p2_data - p1_data).half()  # store diff in half to save memory
                self.task_vector_param_dict[n1] = diff

                # 明示的なメモリ解放
                torch.cuda.empty_cache()


###############################################################################
# forward 時に (base_param + Σ(λ_i * diff_i)) を合成するラッパークラス
###############################################################################
class MergedModelWrapper(nn.Module):
    def __init__(self, pretrained_model, task_vectors, lambdas, chunk_size=256):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.task_vectors = task_vectors
        self.lambdas = lambdas
        self.chunk_size = chunk_size

        # 8-bitモデルかどうか
        self.is_8bit = hasattr(pretrained_model, 'is_loaded_in_8bit') and pretrained_model.is_loaded_in_8bit

    def forward(self, input_ids, labels=None, **kwargs):
        device = self.lambdas.device
        param_dict = {}

        with torch.set_grad_enabled(True):
            for name, base_param in self.pretrained_model.named_parameters():
                # (8-bit) Skip .to(device).half() if the base is 8-bit 
                if self.is_8bit:
                    # base_param is already in 8-bit on the GPU via bitsandbytes.
                    # We need a float/half version to do the addition. We do
                    # partial chunk-based merges. So first get base_param in FP32:
                    base_gpu = base_param.to(dtype=torch.float16)  # still on GPU in bitsandbytes
                else:
                    base_gpu = base_param.to(device).half()

                merged_param = base_gpu

                # 各タスクの diff をチャンク分割して足す
                for i, tv in enumerate(self.task_vectors):
                    diff_cpu = tv.task_vector_param_dict[name]  # this is half on CPU
                    lam = self.lambdas[i]
                    merged_param = merge_in_chunks(
                        merged_param,   # GPU half
                        diff_cpu,       # CPU half
                        lam,            # GPU float
                        device,
                        self.chunk_size
                    )

                param_dict[name] = merged_param

        outputs = stateless.functional_call(
            self.pretrained_model,
            param_dict,
            (input_ids,),
            {'labels': labels, **kwargs}
        )
        return outputs


def merge_in_chunks(base_gpu, diff_cpu, lam, device, chunk_size=1024):
    """
    base_gpu   : GPU上の [X, ...] shape のテンソル (fp16)
    diff_cpu   : CPU上にある同shapeのテンソル (fp16)
    lam        : GPUのスカラー or テンソル (fp32)
    chunk_size : 分割サイズ
    """
    splitted_base = base_gpu.split(chunk_size, dim=0)
    splitted_diff = diff_cpu.split(chunk_size, dim=0)

    new_chunks = []
    for i, diff_chunk_cpu in enumerate(splitted_diff):
        base_chunk_gpu = splitted_base[i]          # GPU
        diff_chunk_gpu = diff_chunk_cpu.to(device) # CPU -> GPU (1 chunk)

        merged_chunk = base_chunk_gpu + (lam * diff_chunk_gpu)
        new_chunks.append(merged_chunk)

        del diff_chunk_gpu
        torch.cuda.empty_cache()
    
    # VRAMの割り当て状況を確認
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    used_vram = torch.cuda.memory_allocated() / (1024 ** 3)
    free_vram = total_vram - used_vram
    print(f"合計VRAM使用状況: {used_vram:.4f} GB, 空きVRAM: {free_vram:.4f} GB")

    merged_param = torch.cat(new_chunks, dim=0)
    return merged_param


###############################################################################
# メインの LambdaOptimizer クラス (プロンプト部分を -100 ラベルにする)
###############################################################################
class LambdaOptimizerCrossEntropy:
    """
    - モデルパラメータは凍結 (requires_grad=False)
    - λ (self.lambdas) のみ学習 (requires_grad=True)
    - 合成: base_param + sum(lambda_i * diff_i)
    - プロンプト部分を -100, 回答部分を実ラベルにする実装
    """

    def __init__(
        self,
        seed: int,
        pretrained_model,       # transformers の PreTrainedModel (可能性として8bitモデル)
        pretrained_model_name: str,
        tokenizers,             # [tok_gsm8k, tok_mbpp, tok_ja_mgsm]
        models_to_merge: List,  # finetuned models (可8bit)
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
        max_length: int = 512,
    ):
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pretrained_model = pretrained_model
        # 8bitモデルの場合は to(device) 不要
        if not (hasattr(self.pretrained_model, 'is_loaded_in_8bit') and self.pretrained_model.is_loaded_in_8bit):
            self.pretrained_model = self.pretrained_model.to(self.device)

        self.models_to_merge = []
        for model in models_to_merge:
            if not (hasattr(model, 'is_loaded_in_8bit') and model.is_loaded_in_8bit):
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

        # pretrained_modelをCPUに戻す (8bitならGPUのまま)
        if not (hasattr(self.pretrained_model, 'is_loaded_in_8bit') and self.pretrained_model.is_loaded_in_8bit):
            self.pretrained_model = self.pretrained_model.cpu()
            torch.cuda.empty_cache()

        # λ (GPUに配置)
        self.lambdas = torch.tensor(
            initial_lambdas, 
            requires_grad=True, 
            dtype=torch.float32, 
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

        # データ (各10件程度)
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

    ############################################################################
    # (以降) GSM8K/MBPP/ja-MGSMの compute_*_batch_loss
    ############################################################################
    @memory_tracker
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

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels_tensor    = torch.tensor(labels_list,   dtype=torch.long, device=device)

        with torch.set_grad_enabled(True):
            outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
            loss = outputs.loss
            
            if not loss.requires_grad:
                print("Warning: Loss requires_grad is False!")
                loss = loss.detach().requires_grad_(True)
            
            return loss

    @memory_tracker
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

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels_tensor    = torch.tensor(labels_list,   dtype=torch.long, device=device)

        with torch.set_grad_enabled(True):
            outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
            loss = outputs.loss
            
            if not loss.requires_grad:
                print("Warning: Loss requires_grad is False!")
                loss = loss.detach().requires_grad_(True)
            
            return loss

    @memory_tracker
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

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels_tensor    = torch.tensor(labels_list,   dtype=torch.long, device=device)

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

    @memory_tracker
    def train_one_epoch(self, batch_size: int = 2):
        print("\n=== train_one_epoch start ===")
        print(f"Initial λ values: {self.lambdas.detach().cpu().numpy()}")

        # Optimizer 初期化
        self.optimizer.zero_grad()

        # MergedModelWrapper
        merged_model = MergedModelWrapper(
            pretrained_model=self.pretrained_model if hasattr(self.pretrained_model, 'is_loaded_in_8bit') and self.pretrained_model.is_loaded_in_8bit else self.pretrained_model.to(self.device),
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
    @memory_tracker
    def optimize(self):
        best_loss = float("inf")
        best_lambdas = None

        for epoch in range(self.num_epochs):
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
    @memory_tracker
    def optimize_with_optuna(self, n_trials: int = 30) -> np.ndarray:
        def objective(trial):
            # lambda の探索
            lambdas = []
            for i in range(len(self.models_to_merge)):
                val = trial.suggest_float(f'lambda_{i}', 0.0, 1.0)
                lambdas.append(val)
            
            device = next(self.pretrained_model.parameters()).device
            self.lambdas = torch.tensor(lambdas, dtype=torch.float32, device=device, requires_grad=True)

            # 簡易的に1エポックだけ行って loss を返す
            epoch_loss = self.train_one_epoch(batch_size=self.batch_size)
            return epoch_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_lambdas = [study.best_params[f'lambda_{i}'] for i in range(len(self.models_to_merge))]
        best_loss = study.best_value

        # 反映
        device = next(self.pretrained_model.parameters()).device
        self.lambdas = torch.tensor(best_lambdas, dtype=torch.float32, device=device, requires_grad=True)

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
