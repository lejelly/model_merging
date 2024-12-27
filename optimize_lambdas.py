import torch
import json
import logging
import datasets
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from transformers import PreTrainedModel
from vllm import LLM, SamplingParams
from human_eval.data import read_problems
import os
import shutil
import glob
from utils.utils import stream_jsonl, write_jsonl
from tqdm import tqdm
import subprocess
from human_eval.evaluation import evaluate_functional_correctness

from model_merging_methods.task_vector import NewTaskVector
from proposed_methods import metagpt, TaskVectorCalculator
from utils.evaluate_llms_utils import extract_answer_number, extract_answer_number_for_ja_mgsm, batch_data, get_math_task_prompt, generate_code_task_prompt, get_ja_math_task_prompt, compute_score_for_ja_mgsm, read_mbpp
from inference_llms_instruct_math_code import create_llm
from utils.utils import delete_all_models, aggressive_clear_gpu_memory
from utils.utils import LanguageDetector
import random
import re
import copy
import wandb
from transformers import AutoModelForCausalLM
import pandas as pd
from utils.load_config import cache_dir

class LambdaOptimizer:
    def __init__(
        self,
        seed: int,
        pretrained_model: PreTrainedModel,
        pretrained_model_name: str,
        models_to_merge: List[PreTrainedModel],
        tokenizers: List,
        initial_lambdas: Optional[np.ndarray] = None,
        num_epochs: int = 5,
        learning_rate: float = 0.1,
        num_train_samples: int = 10,
        optimizer_type: str = "adam",
        logger: Optional[logging.Logger] = None,
        wandb_project: str = "model_merging",
        wandb_name: Optional[str] = None,
        params_name: Optional[str] = None,
        finetuned_model_names: Optional[List[str]] = None,
        optimized_lambda_filepath: Optional[str] = None
    ):
        self.seed = seed
        self.pretrained_model = pretrained_model
        self.models_to_merge = models_to_merge
        self.tokenizers = tokenizers
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_train_samples = num_train_samples
        self.logger = logger or logging.getLogger(__name__)
        self.pretrained_model_name = pretrained_model_name
        self.params_name = params_name
        self.finetuned_model_names = finetuned_model_names
        self.optimized_lambda_filepath = optimized_lambda_filepath
        
        # 初期λ値の設定
        if initial_lambdas is None:
            initial_lambdas = metagpt(pretrained_model, models_to_merge)
        self.lambdas = torch.tensor(initial_lambdas, requires_grad=True, dtype=torch.float32)
        
        # タスクベクトルの計算
        self.task_vectors = []
        for model in models_to_merge:
            task_vector = NewTaskVector(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                exclude_param_names_regex=[]
            )
            self.task_vectors.append(task_vector)
        
        # オプティマイザの設定
        if optimizer_type.lower() == "adam":
            self.optimizer = torch.optim.Adam([self.lambdas], lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = torch.optim.SGD([self.lambdas], lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # 訓練データの読み込み
        self.gsm8k_data = self._load_gsm8k_train_data()
        self.mbpp_data = self._load_mbpp_train_data()
        self.ja_mgsm_data = self._load_ja_mgsm_train_data()
        
        # wandbの初期化
        self.wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name or f"lambda_opt_seed{seed}_optimizer_{optimizer_type}_num_epochs_{num_epochs}_learning_rate_{learning_rate}_num_train_samples_{num_train_samples}",
            config={
                "seed": seed,
                "pretrained_model": pretrained_model_name,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "num_train_samples": num_train_samples,
                "optimizer_type": optimizer_type,
                "initial_lambdas": initial_lambdas.tolist() if initial_lambdas is not None else None
            }
        )

    def _load_gsm8k_train_data(self) -> List[Dict]:
        """GSM8Kの訓練データを読み込む（ランダムサンプリング）"""
        try:
            dataset = datasets.load_dataset("gsm8k", "main", split="train")
            # データセットをシャッフルしてからサンプリング
            shuffled_dataset = dataset.shuffle(seed=self.seed)
            return shuffled_dataset.select(range(min(self.num_train_samples, len(dataset))))
        except Exception as e:
            self.logger.error(f"Failed to load GSM8K data: {e}")
            return []

    def _load_mbpp_train_data(self) -> List[Dict]:
        """MBPPの訓練データを読み込む（ランダムサンプリング）"""
        try:
            problems = read_mbpp("math_code_data/mbpp.train.jsonl")
            data = []
            for task_id, problem in problems.items():
                prompt = f"\n{problem['text']}\nTest examples:"
                if task_id == 493:
                    # テスト例が長すぎるため、関数名のみを含める
                    test_example = problem['test_list'][0]
                    prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
                else:
                    for test_example in problem['test_list']:
                        prompt += f"\n{test_example}"
                data.append({"task_id": task_id, "prompt": prompt})
            
            # リストをランダムにシャッフル
            random.shuffle(data)
            return data[:min(self.num_train_samples, len(data))]
        except Exception as e:
            self.logger.error(f"Failed to load MBPP data: {e}")
            return []

    def _load_ja_mgsm_train_data(self) -> List[Dict]:
        """ja-MGSMの訓練データを読み込む（ランダムサンプリング）"""
        try:
            dataset = datasets.load_dataset("juletxara/mgsm", split="train", name="ja")
            dataset = dataset.select_columns(["question", "answer_number"])
            # データセットをシャッフルしてからサンプリング
            shuffled_dataset = dataset.shuffle(seed=self.seed)
            return shuffled_dataset.select(range(min(self.num_train_samples, len(dataset))))
        except Exception as e:
            self.logger.error(f"Failed to load ja-MGSM data: {e}")
            return []

    def create_merged_model(self) -> Dict[str, torch.Tensor]:
        """λを使用してモデルのパラメータをマージ（計算グラフを維持）"""
        merged_params = {}
        for name, param in self.pretrained_model.named_parameters():
            if name in self.task_vectors[0].task_vector_param_dict:
                merged_param = param
                for tv, lambda_val in zip(self.task_vectors, self.lambdas):
                    merged_param = merged_param + lambda_val * tv.task_vector_param_dict[name]
                merged_params[name] = merged_param
        return merged_params

    def compute_model_score(self, merged_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """マージされたパラメータを使用してモデルのスコアを計算（計算グラフを維持）"""
        torch.cuda.empty_cache()
        
        # モデルのコピーを作成し、明示的にGPUに移動
        model_copy = copy.deepcopy(self.pretrained_model).cuda()
        
        # マージされたパラメータを設定（計算グラフを維持）
        with torch.set_grad_enabled(True):
            for name, param in model_copy.named_parameters():
                if name in merged_params:
                    # パラメータを明示的にGPUに移動
                    param.data.copy_(merged_params[name].cuda())
                    param.requires_grad_(True)
        
        # 各タスクの評価を実行（テンソル演算として実装）
        scores = []
        for eval_func in [self._evaluate_gsm8k, self._evaluate_mbpp, self._evaluate_ja_mgsm]:
            score = eval_func(model_copy)  # 評価関数も計算グラフを維持するように修正が必要
            if isinstance(score, float):
                score = torch.tensor(score, requires_grad=True, device=self.lambdas.device)
            scores.append(score)
        
        return torch.stack(scores)

    def evaluate_model(self, model: PreTrainedModel, task: str) -> float:
        """指定されたタスクでモデルを評価
        Args:
            model: 評価対象のモデル
            task: 評価タスク名（"gs
        """
        if task == "gsm8k":
            return self._evaluate_gsm8k(model)
        elif task == "mbpp":
            return self._evaluate_mbpp(model)
        elif task == "ja_mgsm":
            return self._evaluate_ja_mgsm(model)
        else:
            raise ValueError(f"Unknown task: {task}")

    def _evaluate_gsm8k(self, model: PreTrainedModel) -> torch.Tensor:
        """GSM8Kでの評価（真のバッチ処理版）"""
        batch_size = 20
        
        # プロンプトの準備
        gsm8k_ins = []
        gsm8k_answers = []
        problem_prompt = get_math_task_prompt("zeroshotcot")
        
        for item in self.gsm8k_data:
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item["answer"].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

        # バッチ処理
        all_results = []
        for i in range(0, len(gsm8k_ins), batch_size):
            batch_prompts = gsm8k_ins[i:i + batch_size]
            batch_answers = gsm8k_answers[i:i + batch_size]
            
            # バッチ生成を実行
            generated_texts, outputs = generate_with_gradients_batch(
                model=model,
                tokenizer=self.tokenizers[0],
                prompts=batch_prompts,
                max_length=1024,
                temperature=0.0,
                top_p=1.0
            )
            
            # バッチ内の結果を処理
            for text, answer in zip(generated_texts, batch_answers):
                pred = extract_answer_number(text)
                correct = torch.tensor(
                    float(float(pred) == float(answer)) if pred is not None else 0.0,
                    dtype=torch.float32,
                    device=model.device,
                    requires_grad=True
                )
                all_results.append(correct)
            
            torch.cuda.empty_cache()
        
        results_tensor = torch.stack(all_results)
        return results_tensor.mean()

    def _evaluate_mbpp(self, model: PreTrainedModel) -> torch.Tensor:
        """MBPPでの評価（真のバッチ処理版）"""
        batch_size = 20
        
        # プロンプトの準備
        prompts = []
        task_ids = []
        for item in self.mbpp_data:
            prompt = generate_code_task_prompt(item["prompt"].replace('    ', '\t'))
            prompts.append(prompt)
            task_ids.append(item["task_id"])
        
        # バッチ処理
        completion_seqs = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_task_ids = task_ids[i:i + batch_size]
            
            # バッチ生成を実行
            generated_texts, outputs = generate_with_gradients_batch(
                model=model,
                tokenizer=self.tokenizers[1],
                prompts=batch_prompts,
                max_length=2048,
                temperature=0.0,
                top_p=1.0
            )
            
            # 生成結果の処理
            for text, task_id in zip(generated_texts, batch_task_ids):
                completion_seq = text.split("### Response:")[-1]
                completion_seq = completion_seq.replace('\t', '    ')
                
                completion = self._process_mbpp_completion(completion_seq)
                
                completion_seqs.append({
                    'task_id': task_id,
                    'completion': completion,
                    'all_code': text
                })
            
            torch.cuda.empty_cache()
        
        # 評価用の一時ディレクトリとファイルの設定
        temp_model_path = f"./save_merge_models/temp_merged_model/code/{self.params_name}"
        os.makedirs(temp_model_path, exist_ok=True)
        temp_results_folder = f"./temp_results/mbpp/{self.params_name}"
        os.makedirs(temp_results_folder, exist_ok=True)
        
        # モデルと tokenizer の一時保存
        model.save_pretrained(temp_model_path)
        self.tokenizers[1].save_pretrained(temp_model_path)
        
        # 生成結果をJSONLファイルに保存
        temp_generations_path = f"{temp_results_folder}/generations.jsonl"
        with open(temp_generations_path, "w", encoding="utf-8") as fout:
            for seq in completion_seqs:
                json.dump(seq, fout)
                fout.write("\n")
        
        # MBPPの評価スクリプトを実行
        cmd = [
            "bash", 
            "scripts/exec_eval_mbpp.sh",
            "--model", temp_model_path,
            "--load_generations_path", temp_generations_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout
            accuracy_match = re.search(r'MBPP_ACCURACY=([\d.]+)', output)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
            else:
                self.logger.error("Could not find MBPP_ACCURACY in output")
                accuracy = 0.0
        except Exception as e:
            self.logger.error(f"Error running MBPP evaluation: {e}")
            accuracy = 0.0
        
        # 一時ファイルとディレクトリの削除
        if os.path.exists(temp_model_path):
            for item in os.listdir(temp_model_path):
                item_path = os.path.join(temp_model_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            os.rmdir(temp_model_path)
        shutil.rmtree(temp_results_folder, ignore_errors=True)
        
        # 勾配計算用のテンソルに変換
        accuracy_tensor = torch.tensor(
            accuracy,
            dtype=torch.float32,
            device=model.device,
            requires_grad=True
        )
        
        # GPUメモリのクリーンアップ
        torch.cuda.empty_cache()
        
        return accuracy_tensor

    def _evaluate_ja_mgsm(self, model: PreTrainedModel) -> torch.Tensor:
        """ja-mgsmでの評価（シンプル版）"""
        # 言語検出器の初期化
        lang_detect = LanguageDetector()
        
        # データの確認
        if not self.ja_mgsm_data or len(self.ja_mgsm_data) == 0:
            self.logger.error("No ja-mgsm data available for evaluation")
            return torch.tensor(0.0, dtype=torch.float32, device=model.device, requires_grad=True)
        
        # プロンプトと回答の準備
        questions = []
        prompts = []
        answers = []
        resps = []
        preds = []
        
        try:
            # 各サンプルを個別に処理
            for item in self.ja_mgsm_data:
                questions.append(item["question"])
                prompt = get_ja_math_task_prompt(input_text=item["question"])
                answers.append(item["answer_number"])
                
                # 個別に生成を実行
                generated_text, _ = generate_with_gradients(
                    model=model,
                    tokenizer=self.tokenizers[2],
                    prompt=prompt,
                    max_length=1024,
                    temperature=0.0,
                    top_p=1.0
                )
                
                resps.append(generated_text)
                preds.append(extract_answer_number_for_ja_mgsm(generated_text))
                
                # 各生成後にGPUメモリをクリア
                torch.cuda.empty_cache()
            
            # 結果の処理
            results = {
                "question": questions,
                "answer": answers,
                "response": resps,
                "prediction": preds,
            }
            
            res_dict, incorrect, incorrects_ja, all_response = compute_score_for_ja_mgsm(results, lang_detect)
            accuracy = res_dict.get("acc_ja", 0.0)
            
        except Exception as e:
            self.logger.error(f"Error in ja-mgsm evaluation: {str(e)}")
            accuracy = 0.0
        
        # 精度をテンソルに変換（勾配計算用）
        accuracy_tensor = torch.tensor(
            accuracy, 
            dtype=torch.float32, 
            device=model.device, 
            requires_grad=True
        )
        
        # GPUメモリのクリーンアップ
        torch.cuda.empty_cache()
        
        return accuracy_tensor

    def optimize(self) -> np.ndarray:
        """λ値を最適化（計算グラフを維持）"""
        try:
            best_lambdas = None
            best_loss = float('inf')
            
            # モデルを評価モードに設定
            self.pretrained_model.eval()
            
            lambda_str = "[" + ", ".join([f"{l}" for l in self.lambdas]) + "]"
            self.logger.info(
                f"Epoch 0 | "
                f"Initial Lambdas: {lambda_str}"
            )
            
            for epoch in range(self.num_epochs):
                epoch_start_time = time.time()
                
                # 最適化ステップ
                self.optimizer.zero_grad()
                merged_params = self.create_merged_model()
                scores = self.compute_model_score(merged_params)
                
                # 調和平均の計算（計算グラフを維持）
                n = torch.tensor(len(scores), dtype=torch.float32, device=scores.device, requires_grad=True)
                eps = 1e-8
                denominator = torch.sum(1.0 / (scores + eps))
                harmonic_mean = n / denominator
                
                # 損失の計算
                task_loss = 1.0 - harmonic_mean
                lambda_regularization = 0.01 * torch.sum(self.lambdas ** 2)
                loss = task_loss + lambda_regularization
                
                # 勾配の計算と更新
                loss.backward()
                self.optimizer.step()
                
                # 現在の状態のロギング
                with torch.no_grad():
                    lambda_str = "[" + ", ".join([f"{l}" for l in self.lambdas]) + "]"
                    self.logger.info(
                        f"Epoch {epoch+1} | Loss: {loss.item()} | "
                        f"GSM8K: {scores[0].item()} | MBPP: {scores[1].item()} | "
                        f"JA-MGSM: {scores[2].item()} | Lambdas: {lambda_str}"
                    )
                    
                    # wandbへのログ記録
                    self.wandb_run.log({
                        "epoch": epoch,
                        "loss": loss.item(),
                        "gsm8k_score": scores[0].item(),
                        "mbpp_score": scores[1].item(),
                        "ja_mgsm_score": scores[2].item(),
                        "harmonic_mean": harmonic_mean.item(),
                        **{f"lambda_{i}": l.item() for i, l in enumerate(self.lambdas)}
                    })
                
                # ベストモデルの保存
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_lambdas = self.lambdas.clone()
                    self.logger.info(f"New best loss: {best_loss:.4f} with lambdas: {lambda_str}")
                    
                    # optimized lambdasの保存（追記モード）
                    optimized_df = pd.DataFrame({
                        'epoch': [epoch + 1],
                        'model_name': [self.finetuned_model_names],
                        'lambdas': [best_lambdas.detach().cpu().numpy().tolist()],
                        'loss': [best_loss],
                        'gsm8k_score': [scores[0].item()],
                        'mbpp_score': [scores[1].item()],
                        'ja_mgsm_score': [scores[2].item()]
                    })
                    
                    # ファイルが存在する場合は追記、存在しない場合は新規作成
                    if os.path.exists(self.optimized_lambda_filepath):
                        optimized_df.to_csv(self.optimized_lambda_filepath, mode='a', header=False, index=False)
                    else:
                        optimized_df.to_csv(self.optimized_lambda_filepath, index=False)
                
                # メモリの解放
                del scores, loss, harmonic_mean, merged_params
                torch.cuda.empty_cache()
                
                epoch_end_time = time.time()
                self.logger.info(f"Epoch {epoch+1} took {epoch_end_time - epoch_start_time:.2f} seconds")

        finally:
            pass

        self.wandb_run.finish()
        return best_lambdas

    def _process_mbpp_completion(self, completion: str) -> str:
        """MBPPの生成結果を処理する補助関数"""
        if '```python' in completion:
            completion = completion[completion.index('```python'):].strip()
            completion = completion.replace('```python', '')
            try:
                completion = completion[:completion.index('```')].strip()
            except:
                pass
        
        # その他の処理も同様に実装
        patterns = [
            ('__name__ == "__main__"', 'if __name__ == "__main__":'),
            ('# Example usage', '# Example usage'),
            ('# Test examples', '# Test examples'),
            ('The solution is:', 'The solution is:'),
            ('The answer is:', 'The answer is:')
        ]
        
        for _, pattern in patterns:
            if pattern in completion:
                completion = completion[:completion.index(pattern)].strip()
        
        return completion


def generate_with_gradients(
    model: PreTrainedModel,
    tokenizer,
    prompt: str,
    max_length: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "cuda"
) -> str:
    """勾配計算が可能な生成関数
    
    Args:
        model: 生成に使用するモデル
        tokenizer: トークナイザー
        prompt: 入力プロンプト
        max_length: 最大生成長
        temperature: 生成時の温度
        top_p: top-p サンプリングの閾値
        device: 使用するデバイス
    
    Returns:
        str: 生成されたテキスト
    """
    # 入力のエンコード
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    # 明示的にすべての入力をGPUに移動
    inputs = {k: v.cuda() for k, v in inputs.items()}

    # 勾配計算を有効にして生成
    with torch.set_grad_enabled(True):
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    # 生成されたテキストをデコード
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    return generated_text, outputs

def generate_with_gradients_batch(
    model: PreTrainedModel,
    tokenizer,
    prompts: List[str],
    max_length: int = 1024,
    temperature: float = 0.0,
    top_p: float = 1.0,
    device: str = "cuda"
) -> Tuple[List[str], Any]:
    """バッチ処理による生成関数
    
    Args:
        model: 生成に使用するモデル
        tokenizer: トークナイザー
        prompts: 入力プロンプトのリスト
        max_length: 最大生成長
        temperature: 生成時の温度
        top_p: top-p サンプリングの閾値
        device: 使用するデバイス
    
    Returns:
        List[str]: 生成されたテキストのリスト
        Any: モデルの出力
    """
    # バッチ入力のエンコード
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(device)

    # バッチ生成
    with torch.set_grad_enabled(True):
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )
    
    # バッチ出力のデコード
    generated_texts = tokenizer.batch_decode(
        outputs.sequences, 
        skip_special_tokens=True
    )
    
    return generated_texts, outputs


