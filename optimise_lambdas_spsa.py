import torch
import json
import logging
import datasets
from typing import List, Dict, Optional
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


class LambdaOptimizerSPSA:
    def __init__(
        self,
        seed: int,
        pretrained_model: PreTrainedModel,
        pretrained_model_name: str,
        models_to_merge: List[PreTrainedModel],
        tokenizers: List,
        initial_lambdas: Optional[np.ndarray] = None,
        num_steps: int = 5,
        learning_rate: float = 0.01,
        epsilon: float = 1e-3,
        num_train_samples: int = 10,
        logger: Optional[logging.Logger] = None,
        wandb_project: str = "model_merging",
        wandb_name: Optional[str] = None,
        params_name: Optional[str] = None
    ):
        self.seed = seed
        self.pretrained_model = pretrained_model
        self.models_to_merge = models_to_merge
        self.tokenizers = tokenizers
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_train_samples = num_train_samples
        self.logger = logger or logging.getLogger(__name__)
        self.pretrained_model_name = pretrained_model_name
        self.params_name = params_name
        
        # 初期λ値の設定
        if initial_lambdas is None:
            initial_lambdas = metagpt(pretrained_model, models_to_merge)
        self.lambdas = torch.tensor(initial_lambdas, dtype=torch.float32)
        
        # タスクベクトルの計算
        self.task_vectors = []
        for model in models_to_merge:
            task_vector = NewTaskVector(
                pretrained_model=pretrained_model,
                finetuned_model=model,
                exclude_param_names_regex=[]
            )
            self.task_vectors.append(task_vector)
        
        # 訓練データの読み込み
        self.gsm8k_data = self._load_gsm8k_train_data()
        self.mbpp_data = self._load_mbpp_train_data()
        self.ja_mgsm_data = self._load_ja_mgsm_train_data()
        
        # wandbの初期化
        self.wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name or f"lambda_opt_spsa_seed{seed}_num_steps_{num_steps}_learning_rate_{learning_rate}_num_train_samples_{num_train_samples}",
            config={
                "seed": seed,
                "pretrained_model": pretrained_model_name,
                "num_steps": num_steps,
                "learning_rate": learning_rate,
                "epsilon": epsilon,
                "num_train_samples": num_train_samples,
                "initial_lambdas": initial_lambdas.tolist() if initial_lambdas is not None else None
            }
        )

    def _load_gsm8k_train_data(self) -> List[Dict]:
        """GSM8Kの訓練データを読み込む（ランダムサンプリング）"""
        try:
            dataset = datasets.load_dataset("gsm8k", "main", split="train")
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
                    test_example = problem['test_list'][0]
                    prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
                else:
                    for test_example in problem['test_list']:
                        prompt += f"\n{test_example}"
                data.append({"task_id": task_id, "prompt": prompt})
            
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
            shuffled_dataset = dataset.shuffle(seed=self.seed)
            return shuffled_dataset.select(range(min(self.num_train_samples, len(dataset))))
        except Exception as e:
            self.logger.error(f"Failed to load ja-MGSM data: {e}")
            return []

    def create_merged_model(self) -> Dict[str, torch.Tensor]:
        """λを使用してモデルのパラメータをマージ（メモリ効率化版）"""
        merged_params = {}
        with torch.no_grad():  # 勾配計算を無効化
            for name, param in self.pretrained_model.named_parameters():
                if name in self.task_vectors[0].task_vector_param_dict:
                    # 1回だけcloneを作成
                    merged_param = param.clone()
                    
                    # インプレース演算で更新
                    for tv, lambda_val in zip(self.task_vectors, self.lambdas):
                        merged_param.add_(tv.task_vector_param_dict[name] * lambda_val)
                    
                    merged_params[name] = merged_param
                    
                    # 必要に応じてGPUメモリをクリア
                    torch.cuda.empty_cache()
        
        return merged_params

    def compute_model_score(self, merged_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """マージされたパラメータを使用してモデルのスコアを計算"""
        # deepcopyを避ける
        original_params = {}
        with torch.no_grad():
            for name, param in self.pretrained_model.named_parameters():
                if name in merged_params:
                    original_params[name] = param.data.clone()
                    param.data.copy_(merged_params[name])
        
        # 評価実行
        scores = torch.tensor([
            self._evaluate_gsm8k(self.pretrained_model),
            self._evaluate_mbpp(self.pretrained_model),
            self._evaluate_ja_mgsm(self.pretrained_model)
        ])
        
        # 元のパラメータに戻す
        with torch.no_grad():
            for name, param in self.pretrained_model.named_parameters():
                if name in original_params:
                    param.data.copy_(original_params[name])
        
        return scores

    def _evaluate_gsm8k(self, model: PreTrainedModel) -> float:
        """GSM8Kでの評価"""
        delete_all_models()
        aggressive_clear_gpu_memory()
        
        temp_model_path = f"./save_merge_models/temp_merged_model/math/{self.params_name}"
        os.makedirs(temp_model_path, exist_ok=True)
        model.save_pretrained(temp_model_path)
        self.tokenizers[0].save_pretrained(temp_model_path)
        
        llm = create_llm(
            finetuned_model_name=temp_model_path,
            pretrained_model_name=self.pretrained_model_name,
            args=None,
            logger=self.logger,
            tensor_parallel_size=1,
            just_inference=True
        )
        
        gsm8k_ins = []
        gsm8k_answers = []
        problem_prompt = get_math_task_prompt("zeroshotcot")
        
        for item in self.gsm8k_data:
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item["answer"].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)
        
        batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=60)
        
        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=1024, stop=stop_tokens)
        
        res_completions = []
        for prompt in batch_gsm8k_ins:
            if isinstance(prompt, list):
                pass
            else:
                prompt = [prompt]
            completions = llm.generate(prompt, sampling_params)
            for output in completions:
                generated_text = output.outputs[0].text
                res_completions.append(generated_text)
        
        results = []
        invalid_outputs = []
        for prompt, completion, prompt_answer in zip(gsm8k_ins, res_completions, gsm8k_answers):
            y_pred = extract_answer_number(completion)
            if y_pred is not None:
                results.append(float(y_pred) == float(prompt_answer))
            else:
                results.append(False)
                temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
                invalid_outputs.append(temp)
        
        accuracy = sum(results) / len(results)
        del llm
        torch.cuda.empty_cache()
        if os.path.exists(temp_model_path):
            for item in os.listdir(temp_model_path):
                item_path = os.path.join(temp_model_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            os.rmdir(temp_model_path)
            
        return accuracy

    def _evaluate_mbpp(self, model: PreTrainedModel) -> float:
        """MBPPでの評価"""
        delete_all_models()
        aggressive_clear_gpu_memory()
        
        temp_model_path = f"./save_merge_models/temp_merged_model/code/{self.params_name}"
        os.makedirs(temp_model_path, exist_ok=True)
        temp_results_folder = f"./temp_results/mbpp/{self.params_name}"
        os.makedirs(temp_results_folder, exist_ok=True)
        
        model.save_pretrained(temp_model_path)
        self.tokenizers[1].save_pretrained(temp_model_path)
        
        llm = create_llm(
            finetuned_model_name=temp_model_path,
            pretrained_model_name=self.pretrained_model_name,
            args=None,
            logger=self.logger,
            tensor_parallel_size=1,
            just_inference=True
        )
        
        BATCH_SIZE = 50
        
        prompts = []
        task_ids = []
        for item in self.mbpp_data:
            prompt = item["prompt"].replace('    ', '\t')
            prompts.append(prompt)
            task_ids.append(item["task_id"])
        
        prompt_batches = [prompts[i:i + BATCH_SIZE] for i in range(0, len(prompts), BATCH_SIZE)]
        task_id_batches = [task_ids[i:i + BATCH_SIZE] for i in range(0, len(task_ids), BATCH_SIZE)]
        
        sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)
        
        completion_seqs = []
        for prompt_batch, ids_batch in zip(prompt_batches, task_id_batches):
            batch_prompts = [generate_code_task_prompt(p) for p in prompt_batch]
            
            completions = llm.generate(batch_prompts, sampling_params)
            
            for task_id, completion in zip(ids_batch, completions):
                gen_seq = completion.outputs[0].text
                completion_seq = gen_seq.split("### Response:")[-1]
                completion_seq = completion_seq.replace('\t', '    ')
                all_code = gen_seq.replace('\t', '    ')
                
                completion = completion_seq.strip()
                if '```python' in completion:
                    def_line = completion.index('```python')
                    completion = completion[def_line:].strip()
                    completion = completion.replace('```python', '')
                    try:
                        next_line = completion.index('```')
                        completion = completion[:next_line].strip()
                    except:
                        self.logger.info("wrong completion")
                
                if "__name__ == \"__main__\"" in completion:
                    try:
                        next_line = completion.index('if __name__ == "__main__":')
                        completion = completion[:next_line].strip()
                    except:
                        self.logger.info("wrong completion")
                
                if "# Example usage" in completion:
                    next_line = completion.index('# Example usage')
                    completion = completion[:next_line].strip()
                
                if "# Test examples" in completion:
                    next_line = completion.index('# Test examples')
                    completion = completion[:next_line].strip()
                
                if "The solution is:" in completion:
                    def_line = completion.index("The solution is:")
                    completion = completion[def_line:].strip()
                    completion = completion.replace('The solution is:', '')
                    try:
                        next_line = completion.index('\n\nThe answer is:')
                        completion = completion[:next_line].strip()
                    except:
                        completion = completion.strip()
                        self.logger.info("maybe wrong completion")
                
                if "The answer is:" in completion:
                    def_line = completion.index("The answer is:")
                    completion = completion[def_line:].strip()
                    completion = completion.replace('The answer is:', '')
                    try:
                        next_line = completion.index('\n\nThe answer is:')
                        completion = completion[:next_line].strip()
                    except:
                        completion = completion.strip()
                        self.logger.info("maybe wrong completion")
                
                completion_seqs.append({
                    'task_id': task_id,
                    'completion': completion,
                    'all_code': all_code
                })
        
        temp_generations_path = f"{temp_results_folder}/generations.jsonl"
        with open(temp_generations_path, "w", encoding="utf-8") as fout:
            for seq in completion_seqs:
                json.dump(seq, fout)
                fout.write("\n")
        
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
        
        del llm
        torch.cuda.empty_cache()
        if os.path.exists(temp_model_path):
            for item in os.listdir(temp_model_path):
                item_path = os.path.join(temp_model_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            os.rmdir(temp_model_path)
        shutil.rmtree(temp_results_folder, ignore_errors=True)
        
        return accuracy

    def _evaluate_ja_mgsm(self, model: PreTrainedModel) -> float:
        """ja-mgsmでの評価"""
        delete_all_models()
        aggressive_clear_gpu_memory()
        
        temp_model_path = f"./save_merge_models/temp_merged_model/jp/{self.params_name}"
        os.makedirs(temp_model_path, exist_ok=True)
        model.save_pretrained(temp_model_path)
        self.tokenizers[2].save_pretrained(temp_model_path)
        
        llm = create_llm(
            finetuned_model_name=temp_model_path,
            pretrained_model_name=self.pretrained_model_name,
            args=None,
            logger=self.logger,
            tensor_parallel_size=1,
            just_inference=True
        )
        
        lang_detect = LanguageDetector()
        
        questions = []
        prompts = []
        answers = []
        for item in self.ja_mgsm_data:
            questions.append(item["question"])
            prompts.append(get_ja_math_task_prompt(input_text=item["question"]))
            answers.append(item["answer_number"])
        
        stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
        sampling_params = SamplingParams(
            temperature=0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_tokens=1024,
            stop=stop_tokens
        )
        
        with torch.no_grad():
            outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
        
        resps = [output.outputs[0].text for output in outputs]
        preds = [extract_answer_number_for_ja_mgsm(t) for t in resps]
        
        results = {
            "question": questions,
            "answer": answers,
            "response": resps,
            "prediction": preds,
        }
        
        res_dict, incorrect, incorrects_ja, _ = compute_score_for_ja_mgsm(results, lang_detect)
        accuracy = res_dict["acc_ja"]
        
        del llm
        torch.cuda.empty_cache()
        if os.path.exists(temp_model_path):
            for item in os.listdir(temp_model_path):
                item_path = os.path.join(temp_model_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
            os.rmdir(temp_model_path)
        
        return accuracy

    def spsa_grad(self, x: np.ndarray) -> np.ndarray:
        """SPSAによる勾配推定"""
        d = len(x)
        # シード固定して乱数生成
        rng = np.random.RandomState(self.seed)
        delta = rng.choice([-1, 1], size=d)
        
        # 摂動を加えたλでのスコア計算
        x_plus = x + self.epsilon * delta
        x_minus = x - self.epsilon * delta
        
        # λを使用してモデルをマージし、評価
        self.lambdas = torch.tensor(x_plus, dtype=torch.float32)
        merged_params_plus = self.create_merged_model()
        scores_plus = self.compute_model_score(merged_params_plus)
        f_plus = torch.mean(scores_plus).item()  # スコアの平均を使用
        
        # メモリ解放
        del merged_params_plus, scores_plus
        torch.cuda.empty_cache()
        
        self.lambdas = torch.tensor(x_minus, dtype=torch.float32)
        merged_params_minus = self.create_merged_model()
        scores_minus = self.compute_model_score(merged_params_minus)
        f_minus = torch.mean(scores_minus).item()
        
        # メモリ解放
        del merged_params_minus, scores_minus
        torch.cuda.empty_cache()
        
        # 勾配推定
        grad_est = (f_plus - f_minus) / (2.0 * self.epsilon) * (1.0 / delta)
        return grad_est

    def compute_harmonic_mean_score(self, scores: torch.Tensor) -> float:
        """
        タスク間のバランスを考慮した調和平均スコアを計算
        Args:
            scores: [gsm8k_score, mbpp_score, ja_mgsm_score]
        Returns:
            float: 調和平均スコア
        """
        # 調和平均の計算
        # これにより、1つのタスクが極端に悪い場合にペナルティを与える
        harmonic_mean = len(scores) / torch.sum(1.0 / (scores + 1e-8))
        return harmonic_mean.item()

    def optimize(self) -> np.ndarray:
        """SPSAによるλ値の最適化"""
        # 初期λ値の設定
        x = self.lambdas.detach().numpy()
        best_x = x.copy()
        best_score = float('-inf')
        
        for step in range(self.num_steps):
            # GPUメモリのクリア
            torch.cuda.empty_cache()
            
            # 勾配推定と更新
            grad_est = self.spsa_grad(x)
            x = x + self.learning_rate * grad_est
            
            # 現在のλでのスコア計算
            self.lambdas = torch.tensor(x, dtype=torch.float32)
            merged_params = self.create_merged_model()
            scores = self.compute_model_score(merged_params)
            current_score = self.compute_harmonic_mean_score(scores)
            
            # ベストスコアの更新
            if current_score > best_score:
                best_score = current_score
                best_x = x.copy()
                self.logger.info(f"New best score: {best_score:.4f} with lambdas: {best_x}")
            
            # 進捗の記録
            self.wandb_run.log({
                "step": step,
                "current_score": current_score,
                "best_score": best_score,
                "gsm8k_score": scores[0].item(),
                "mbpp_score": scores[1].item(),
                "ja_mgsm_score": scores[2].item(),
                **{f"lambda_{i}": l for i, l in enumerate(x)}
            })
            
            # 進捗の表示
            lambda_str = "[" + ", ".join([f"{l:.3f}" for l in x]) + "]"
            self.logger.info(
                f"Step {step+1:^6d} | Score: {current_score:8.4f} | "
                f"GSM8K: {scores[0]:8.4f} | MBPP: {scores[1]:8.4f} | "
                f"ja-MGSM: {scores[2]:8.4f} | Lambdas: {lambda_str}"
            )
            
            # メモリクリア
            torch.cuda.empty_cache()
        
        # 最終結果の出力
        self.logger.info("Optimization Complete")
        self.logger.info(f"Best Score: {best_score:.4f}")
        self.logger.info(f"Best Lambdas: {[f'{l:.4f}' for l in best_x]}")
        self.logger.info("="*80 + "\n")
        
        # wandbセッションの終了
        self.wandb_run.finish()
        return best_x