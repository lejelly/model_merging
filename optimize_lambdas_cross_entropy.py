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

# Mixed Precision (任意)
from torch.cuda.amp import autocast, GradScaler

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
    例: pretrained_model と finetuned_model の差分をとった辞書
    'param_name' -> Tensor(差分)
    """
    def __init__(self, pretrained_model, finetuned_model, exclude_param_names_regex=[]):
        self.task_vector_param_dict = {}
        with torch.no_grad():  # 勾配計算不要
            for (n1, p1), (n2, p2) in zip(pretrained_model.named_parameters(), finetuned_model.named_parameters()):
                # CPU上で差分を計算し、float32からfloat16に変換してメモリを節約
                diff = (p2.cpu() - p1.cpu()).half()
                self.task_vector_param_dict[n1] = diff
                
                # 明示的なメモリ解放は行わない
                torch.cuda.empty_cache()


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
        pretrained_model,       # transformers の PreTrainedModel
        pretrained_model_name: str,
        tokenizers,             # [tok_gsm8k, tok_mbpp, tok_ja_mgsm]
        models_to_merge: List,  # finetuned models
        initial_lambdas: Optional[np.ndarray] = None,
        num_epochs: int = 3,
        learning_rate: float = 0.001,
        batch_size: int = 10,
        num_train_samples: int = 10,
        optimizer_type: str = "adam",
        logger: Optional[logging.Logger] = None,
        params_name: str = None,
        finetuned_model_names: List[str] = None,
        optimized_lambda_filepath: str = None,
        use_mixed_precision: bool = True,
    ):
        self.seed = seed
        self.logger = logger or logging.getLogger(__name__)
        self.pretrained_model = pretrained_model
        self.pretrained_model_name = pretrained_model_name
        self.tokenizers = tokenizers  # [tok_gsm8k, tok_mbpp, tok_ja_mgsm]
        self.models_to_merge = models_to_merge
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_train_samples = num_train_samples
        self.use_mixed_precision = use_mixed_precision

        # λ の初期値
        if initial_lambdas is None:
            initial_lambdas = np.zeros(len(models_to_merge))

        # モデルとTask Vectorsを最初からGPUに配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pretrained_model = pretrained_model.to(self.device)
        
        # Task Vectorsの初期化（1回だけ）
        self.task_vectors = []
        for ft_model in models_to_merge:
            ft_model.to(self.device)
            tv = NewTaskVector(pretrained_model, ft_model)
            self.task_vectors.append(tv)
        
        # λもGPUに配置
        self.lambdas = torch.tensor(initial_lambdas, requires_grad=True, 
                                   dtype=torch.float32, device=self.device)

        # 事前学習モデルのパラメータを凍結
        for p in pretrained_model.parameters():
            p.requires_grad = False

        # Optimizer: λ のみ対象
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam([self.lambdas], lr=self.learning_rate)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD([self.lambdas], lr=self.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

        # Mixed Precision
        self.scaler = GradScaler() if self.use_mixed_precision else None

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
                "use_mixed_precision": use_mixed_precision
            }
        )

        # 追加のパラメータ
        self.params_name = params_name
        self.finetuned_model_names = finetuned_model_names
        self.optimized_lambda_filepath = optimized_lambda_filepath

    ############################################################################
    # ★ functional_call を使わずに、都度マージ済みモデルを作成する関数を定義
    ############################################################################
    def create_merged_model(self) -> torch.nn.Module:
        """
        self.pretrained_model のコピーを作成し、
        そこに (ベースパラメータ + Σ(λ_i * diff_i)) を書き込む。
        
        - オリジナルの self.pretrained_model は壊さない
        - コピー先への .copy_() は行うが、あくまで"一時モデルを上書き"するだけなので
          「元モデルを in-place で壊す」わけではない
        - これにより "パラメータが λ に依存" したモデルが得られるので、
          forward → backward すれば λ に勾配が伝わる
        """
        merged_model = copy.deepcopy(self.pretrained_model)
        
        for name, param_m in merged_model.named_parameters():
            base_param = dict(self.pretrained_model.named_parameters())[name]
            diff_sum = torch.zeros_like(base_param, requires_grad=True)
            
            for i, tv in enumerate(self.task_vectors):
                # 必要な時にGPUに移動し、float32に戻す
                diff = tv.task_vector_param_dict[name].to(self.device).float()
                diff_sum = diff_sum + self.lambdas[i] * diff
            
            with torch.no_grad():
                param_m.zero_()
            param_m += base_param + diff_sum
            param_m.requires_grad_(True)

        return merged_model

    ############################################################################
    # (以降) GSM8K/MBPP/ja-MGSMの compute_*_batch_loss: 
    # functional_call をやめて普通に merged_model(...) する
    ############################################################################
    def compute_gsm8k_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model) -> torch.Tensor:
        device = next(self.pretrained_model.parameters()).device

        prompts, answers = [], []
        for item in batch_data:
            p = f"Q: {item['question']}\nA:"
            a = item["answer"]
            prompts.append(p)
            answers.append(a)

        prompt_enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        answer_enc = tokenizer(answers, padding=True, truncation=True, return_tensors="pt").to(device)

        pad_id = tokenizer.pad_token_id or 0

        input_ids_list = []
        labels_list = []
        max_len = 0

        for i in range(len(prompts)):
            p_ids = prompt_enc.input_ids[i].tolist()
            a_ids = answer_enc.input_ids[i].tolist()

            p_len = sum(1 for x in p_ids if x != pad_id)
            a_len = sum(1 for x in a_ids if x != pad_id)

            merged_ids = p_ids[:p_len] + a_ids[:a_len]
            merged_labels = ([-100] * p_len) + a_ids[:a_len]

            max_len = max(max_len, len(merged_ids))
            input_ids_list.append(merged_ids)
            labels_list.append(merged_labels)

        # パディング
        for i in range(len(input_ids_list)):
            diff_len = max_len - len(input_ids_list[i])
            input_ids_list[i].extend([pad_id] * diff_len)
            labels_list[i].extend([-100] * diff_len)

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long, device=device)

        outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
        loss = outputs.loss
        return loss

    def compute_mbpp_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model) -> torch.Tensor:
        device = next(self.pretrained_model.parameters()).device

        prompts, codes = [], []
        for item in batch_data:
            prompts.append(f"{item['text']}\nSolution:\n")
            codes.append(item["code"])

        prompt_enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        code_enc   = tokenizer(codes,   padding=True, truncation=True, return_tensors="pt").to(device)

        pad_id = tokenizer.pad_token_id or 0

        input_ids_list = []
        labels_list = []
        max_len = 0

        for i in range(len(prompts)):
            p_ids = prompt_enc.input_ids[i].tolist()
            c_ids = code_enc.input_ids[i].tolist()

            p_len = sum(1 for x in p_ids if x != pad_id)
            c_len = sum(1 for x in c_ids if x != pad_id)

            merged_ids = p_ids[:p_len] + c_ids[:c_len]
            merged_labels = ([-100] * p_len) + c_ids[:c_len]

            max_len = max(max_len, len(merged_ids))
            input_ids_list.append(merged_ids)
            labels_list.append(merged_labels)

        # パディング
        for i in range(len(input_ids_list)):
            diff_len = max_len - len(input_ids_list[i])
            input_ids_list[i].extend([pad_id] * diff_len)
            labels_list[i].extend([-100] * diff_len)

        input_ids_tensor = torch.tensor(input_ids_list, dtype=torch.long, device=device)
        labels_tensor    = torch.tensor(labels_list,   dtype=torch.long, device=device)

        outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
        loss = outputs.loss
        return loss

    def compute_ja_mgsm_batch_loss(self, batch_data: List[Dict], tokenizer, merged_model) -> torch.Tensor:
        device = next(self.pretrained_model.parameters()).device

        prompts, answers = [], []
        for item in batch_data:
            prompts.append(f"{item['question']}\n答えを考える: ")
            answers.append(item["answer"])

        prompt_enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(device)
        answer_enc = tokenizer(answers, padding=True, truncation=True, return_tensors="pt").to(device)

        pad_id = tokenizer.pad_token_id or 0

        input_ids_list = []
        labels_list = []
        max_len = 0

        for i in range(len(prompts)):
            p_ids = prompt_enc.input_ids[i].tolist()
            a_ids = answer_enc.input_ids[i].tolist()

            p_len = sum(1 for x in p_ids if x != pad_id)
            a_len = sum(1 for x in a_ids if x != pad_id)

            merged_ids = p_ids[:p_len] + a_ids[:a_len]
            merged_labels = ([-100] * p_len) + a_ids[:a_len]

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

        outputs = merged_model(input_ids=input_ids_tensor, labels=labels_tensor)
        loss = outputs.loss
        return loss

    ###########################################################################
    # エポック学習処理 (train_one_epoch)
    ###########################################################################
    def train_one_epoch(self, epoch_idx: int, batch_size: int = 2):
        device = next(self.pretrained_model.parameters()).device
        
        # エポックの開始時に1回だけマージ済みモデルを作成
        merged_model = self.create_merged_model().to(device)
        
        # バッチ処理は同じ
        def make_batches(data_list, bsz):
            random.shuffle(data_list)
            return [data_list[i:i+bsz] for i in range(0, len(data_list), bsz)]

        gsm8k_batches   = make_batches(self.gsm8k_data, batch_size)
        mbpp_batches    = make_batches(self.mbpp_data, batch_size)
        ja_mgsm_batches = make_batches(self.ja_mgsm_data, batch_size)

        epoch_loss = 0.0
        step_count = 0

        # 各compute_*_batch_lossメソッドを修正して、merged_modelを引数として受け取るように変更
        def compute_batch_losses(batch_data, tokenizer, task_type):
            device = next(self.pretrained_model.parameters()).device
            
            if task_type == "gsm8k":
                return self.compute_gsm8k_batch_loss(batch_data, tokenizer, merged_model)
            elif task_type == "mbpp":
                return self.compute_mbpp_batch_loss(batch_data, tokenizer, merged_model)
            else:  # ja_mgsm
                return self.compute_ja_mgsm_batch_loss(batch_data, tokenizer, merged_model)

        # GSM8K
        for batch_data in gsm8k_batches:
            self.optimizer.zero_grad()
            if self.use_mixed_precision:
                with autocast():
                    loss = compute_batch_losses(batch_data, self.tokenizers[0], "gsm8k")
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = compute_batch_losses(batch_data, self.tokenizers[0], "gsm8k")
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            step_count += 1

        # MBPP
        for batch_data in mbpp_batches:
            self.optimizer.zero_grad()
            if self.use_mixed_precision:
                with autocast():
                    loss = compute_batch_losses(batch_data, self.tokenizers[1], "mbpp")
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = compute_batch_losses(batch_data, self.tokenizers[1], "mbpp")
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            step_count += 1

        # ja-MGSM
        for batch_data in ja_mgsm_batches:
            self.optimizer.zero_grad()
            if self.use_mixed_precision:
                with autocast():
                    loss = compute_batch_losses(batch_data, self.tokenizers[2], "ja_mgsm")
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss = compute_batch_losses(batch_data, self.tokenizers[2], "ja_mgsm")
                loss.backward()
                self.optimizer.step()
            
            epoch_loss += loss.item()
            step_count += 1

        if step_count > 0:
            epoch_loss /= step_count
        else:
            epoch_loss = 0.0

        return epoch_loss

    ###########################################################################
    # 通常の学習
    ###########################################################################
    def optimize(self):
        best_loss = float("inf")
        best_lambdas = None

        for epoch in range(self.num_epochs):
            epoch_loss = self.train_one_epoch(epoch, batch_size=self.num_train_samples)
            
            # wandbログ
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

        # ベスト結果のログ
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
            epoch_loss = self.train_one_epoch(0, batch_size=self.num_train_samples)
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
