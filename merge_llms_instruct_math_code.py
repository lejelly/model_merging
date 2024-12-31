import torch
import argparse
import sys
import os
import shutil
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_merging_methods.task_vector import TaskVector
from model_merging_methods.mask_weights_utils import mask_model_weights_exclusive
from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize, copy_params_to_model
from inference_llms_instruct_math_code import create_llm, test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp, test_ja_mgsm
from utils.load_config import cache_dir
from utils.utils import delete_all_models, aggressive_clear_gpu_memory

import numpy as np
from typing import List
from transformers import PreTrainedModel
from tqdm import tqdm

from proposed_methods import metagpt, WeightingStrategy, metagpt_advanced, visualize_task_similarities, calculate_lambda_optimized, calculate_lambda_full, calc_MetaRiemann, metagpt_strict, profile_metagpt_strict, analyze_task_vectors
from calculate_lambdas import calculate_and_save_lambdas

import pandas as pd

from optimize_lambdas import LambdaOptimizer
from optimise_lambdas_spsa import LambdaOptimizerSPSA
from optimize_lambdas_cross_entropy import LambdaOptimizerCrossEntropy
from test_lambda_gradients import test_lambda_gradients

task_model_mapping_dict = {
    "jp1": "augmxnt/shisa-gamma-7b-v1",
    "jp2": "tokyotech-llm/Swallow-MS-7b-v0.1",
    "math1": "WizardLMTeam/WizardMath-7B-V1.1",
    "math2": "GAIR/Abel-7B-002",
    "math3": "upaya07/Arithmo2-Mistral-7B",
    "bio": "BioMistral/BioMistral-7B",
    
    "instruct": "mistralai/Mistral-7B-Instruct-v0.2",
    "math": "TIGER-Lab/MAmmoTH2-7B",
    "code": "Nondzu/Mistral-7B-codealpaca-lora",
    "jp": "augmxnt/shisa-gamma-7b-v1",
    
}
finetuned_model_backbone_mapping_dict = {
    "WizardLMTeam/WizardMath-7B-V1.1": "mistralai/Mistral-7B-v0.1",
    "augmxnt/shisa-gamma-7b-v1": "mistralai/Mistral-7B-v0.1",
    "GAIR/Abel-7B-002": "mistralai/Mistral-7B-v0.1",
    "tokyotech-llm/Swallow-MS-7b-v0.1": "mistralai/Mistral-7B-v0.1",
    "BioMistral/BioMistral-7B": "mistralai/Mistral-7B-v0.1",
    "upaya07/Arithmo2-Mistral-7B": "mistralai/Mistral-7B-v0.1",
    
    "meta-llama/Llama-2-7b-chat-hf": "meta-llama/Llama-2-7b",
    "TIGER-Lab/MAmmoTH-7B": "meta-llama/Llama-2-7b",
    "mrm8488/llama-2-coder-7b": "meta-llama/Llama-2-7b",
    
    "mistralai/Mistral-7B-Instruct-v0.2": "mistralai/Mistral-7B-v0.1",
    "TIGER-Lab/MAmmoTH2-7B": "mistralai/Mistral-7B-v0.1",
    "Nondzu/Mistral-7B-codealpaca-lora": "mistralai/Mistral-7B-v0.1", 
}

def print_lambda_distribution(lambdas, model_names):
    # もしlambdasがPyTorch tensorなら、NumPy配列に変換
    if torch.is_tensor(lambdas):
        lambdas_sum = torch.sum(lambdas).item()
        lambdas_np = lambdas.detach().cpu().numpy()
    else:
        lambdas_sum = np.sum(lambdas)
        lambdas_np = lambdas

    print("\nLambda distribution:")
    for i, (name, lambda_val) in enumerate(zip(model_names, lambdas_np)):
        print(f"{name}: {lambda_val:.6f}")
    print(f"Sum of lambdas: {lambdas_sum:.6f}")


def get_merge_performance(args: argparse.Namespace, finetuned_model_names: list, merge_task_names: list, models_to_merge: list, trainers: list, logger: logging.Logger,
                          merging_method: MergingMethod, tokenizers: list):
    """
    get the performance of merging method named merging_method_name
    :param args: ArgumentParser, input argument parser
    :param finetuned_model_names: list, names of finetuned models
    :param merge_task_names: list, names of tasks that need to be merged
    :param models_to_merge: list, individual models that need to be merged
    :param trainers: list, trainers of individual models
    :param logger: Logger, logger
    :param merging_method: MergingMethod, the mering method
    :param tokenizers: list of tokenizers
    :return:
    """
    
    logger.info(f"configuration is {args}")

    try:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name), device_map="cpu", torch_dtype=torch.bfloat16, load_in_4bit=True)
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name))
        pretrained_model.gradient_checkpointing_enable()
    except:
        if "meta-llama/Llama-2-7b" in args.pretrained_model_name:
            from transformers import LlamaForCausalLM, LlamaTokenizer
            pretrained_model = LlamaForCausalLM.from_pretrained("/work/gb20/b20042/model_merging/llama2")
            pretrained_tokenizer = LlamaTokenizer.from_pretrained("/work/gb20/b20042/model_merging/llama2")
            pretrained_model.gradient_checkpointing_enable()
        else:
            pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir, device_map="cpu", torch_dtype=torch.bfloat16, load_in_4bit=True)
            pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir)
            pretrained_model.gradient_checkpointing_enable()
    
    
    if "GAIR/Abel-7B-002" in finetuned_model_names:
        pad_token = "<extra_id_32001><extra_id_32002><extra_id_32003><extra_id_32004><extra_id_32005><extra_id_32006><extra_id_32007><extra_id_32008><extra_id_32009><extra_id_32010><extra_id_32011><extra_id_32012><extra_id_32013><extra_id_32014><extra_id_32015><extra_id_32016><extra_id_32017><extra_id_32018><extra_id_32019><extra_id_32020><extra_id_32021><extra_id_32022><extra_id_32023><extra_id_32024><extra_id_32025><extra_id_32026><extra_id_32027><extra_id_32028><extra_id_32029><extra_id_32030><extra_id_32031><pad>"
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=pad_token),
            model=pretrained_model,
            tokenizer=pretrained_tokenizer,
        )
        for finetuned_model, finetuned_tokenizer in zip(models_to_merge, tokenizers):
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=pad_token),
                model=finetuned_model,
                tokenizer=finetuned_tokenizer,
            )
    else:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            model=pretrained_model,
            tokenizer=pretrained_tokenizer,
        )
        for finetuned_model, finetuned_tokenizer in zip(models_to_merge, tokenizers):
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                model=finetuned_model,
                tokenizer=finetuned_tokenizer,
            )
    
    # set random seed to guarantee reproducibility
    set_random_seed(seed=args.seed)
    merged_model = pretrained_model
    
    # visualize task cosine similarities
    """
    print(f"Visualizing task cosine similarities...")
    visualizer = visualize_task_similarities(
        pretrained_model=merged_model,
        models=models_to_merge,
        model_names=args.merge_task_names,
        save_dir=f"./figs/{args.lambda_strategy}",
    )
    """
    
    if args.metagpt:
        print("Start calculating lambdas...")
        print(f"Lambda_strategy: {args.lambda_strategy}")        
        
        if args.lambda_strategy == "metagpt":
            optimized_lambdas = metagpt(pretrained_model, models_to_merge)
            print_lambda_distribution(optimized_lambdas, finetuned_model_names)
        elif args.lambda_strategy == "metagpt_strict":
            optimized_lambdas = metagpt_strict(pretrained_model, models_to_merge)
            print_lambda_distribution(optimized_lambdas, finetuned_model_names)
        else:
            # ファイル名の生成
            strategy = WeightingStrategy(args.lambda_strategy)
            params_name = f"{strategy.value}/{args.optimizer_type}_epochs{args.num_epochs}_lr{args.learning_rate}_sample{args.num_train_samples}"
            save_dir = f"./lambdas/seed{args.seed}/{'_'.join(merge_task_names)}/{params_name}" 
            os.makedirs(save_dir, exist_ok=True)
            model_names_str = '_'.join(sorted([name.split('/')[-1] for name in finetuned_model_names]))
            initial_lambda_filename = f'initial_lambdas_{strategy.value}_{model_names_str}.csv'
            optimized_lambda_filename = f'optimized_lambdas_{strategy.value}_{model_names_str}.csv'
   
            if args.initial_lambda_filepath is not None:
                initial_lambda_filepath = args.initial_lambda_filepath
            else:
                initial_lambda_filepath = os.path.join(save_dir, initial_lambda_filename)
            
            if args.optimized_lambda_filepath is not None:
                optimized_lambda_filepath = args.optimized_lambda_filepath
            else:
                optimized_lambda_filepath = os.path.join(save_dir, optimized_lambda_filename)
            
            # 既存のλ値をチェック
            if os.path.exists(initial_lambda_filepath):
                print(f"Loading existing lambdas from: {save_dir}")
                # initial lambdasの読み込み
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
                else:
                    initial_lambdas = metagpt(pretrained_model, models_to_merge)
                    
                # initial lambdasの保存
                if args.lambda_strategy != "optuna":
                    initial_df = pd.DataFrame({
                        'model_name': finetuned_model_names,
                        'lambda': initial_lambdas
                    })
                    initial_df.to_csv(initial_lambda_filepath, index=False)
            
            if os.path.exists(optimized_lambda_filepath):
                # optimized lambdasの読み込み
                optimized_df = pd.read_csv(optimized_lambda_filepath)
                if args.lambda_strategy == "optuna":
                    # lambdasカラムから直接リストを取得
                    lambda_str = optimized_df.loc[0, 'lambdas']  
                    # 文字列からリストに変換
                    optimized_lambdas = eval(lambda_str)
                else:
                    # lossが最小のepochを見つける
                    min_loss_epoch = optimized_df['loss'].idxmin()
                    # そのepochのλ値を取得
                    lambda_str = optimized_df.loc[min_loss_epoch, 'lambdas']
                    # 文字列からリストに変換
                    optimized_lambdas = eval(lambda_str)
            else:
                optimizer = LambdaOptimizerCrossEntropy(
                    seed=args.seed,
                    pretrained_model=pretrained_model,
                    pretrained_model_name=args.pretrained_model_name,
                    models_to_merge=models_to_merge,
                    tokenizers=tokenizers,
                    initial_lambdas=initial_lambdas,
                    num_epochs=args.num_epochs,
                    learning_rate=args.learning_rate,
                    num_train_samples=args.num_train_samples,
                    optimizer_type=args.optimizer_type,
                    logger=logger,
                    params_name=params_name,
                    finetuned_model_names=finetuned_model_names,
                    optimized_lambda_filepath=optimized_lambda_filepath
                )
                
                if args.lambda_strategy == "optuna":
                   optimized_lambdas = optimizer.optimize_with_optuna()
                else:
                   optimized_lambdas = optimizer.optimize()
        
            #print("\nInitial lambdas:")
            #print_lambda_distribution(initial_lambdas, finetuned_model_names)
        print("\nOptimized lambdas:")
        print_lambda_distribution(optimized_lambdas, finetuned_model_names)
        
        # optimized_lambdasの値をargs.gradationに設定
        args.gradation1 = float(optimized_lambdas[0])
        args.gradation2 = float(optimized_lambdas[1])
        args.gradation3 = float(optimized_lambdas[2])
    """        
    if args.single_exclusive_model:
        masked_param_dicts = mask_model_weights_exclusive(finetuned_models=models_to_merge, pretrained_model=merged_model, 
                                                        exclude_param_names_regex=[], weight_format=args.weight_format,
                                                        weight_mask_rate=args.weight_mask_rate,
                                                        use_weight_rescale=args.use_weight_rescale, mask_strategy=args.mask_strategy) 

        copy_params_to_model(params=masked_param_dicts[0], model=models_to_merge[0])
        copy_params_to_model(params=masked_param_dicts[1], model=models_to_merge[1])
        merged_task_vector = TaskVector(pretrained_model=merged_model, finetuned_model=models_to_merge[int(args.subordinate_mask)], exclude_param_names_regex=[])
        merged_params = merged_task_vector.combine_with_pretrained_model(pretrained_model=merged_model, scaling_coefficient=args.scaling_coefficient)
        copy_params_to_model(params=merged_params, model=merged_model)
    else:
        merged_model = merging_method.get_merged_model(merged_model=merged_model,
                                                    models_to_merge=models_to_merge,
                                                    exclude_param_names_regex=[],
                                                    trainers=trainers,
                                                    scaling_coefficient=args.scaling_coefficient,
                                                    nums_fisher_examples=None,
                                                    fisher_scaling_coefficients=None,
                                                    normalize_fisher_weight=None,
                                                    minimal_fisher_weight=None,
                                                    nums_regmean_examples=None,
                                                    reduce_non_diagonal_ratio=None,
                                                    param_value_mask_rate=args.param_value_mask_rate,
                                                    weight_format=args.weight_format,
                                                    weight_mask_rates=args.weight_mask_rates,
                                                    use_weight_rescale=args.use_weight_rescale,
                                                    mask_strategy=args.mask_strategy,
                                                    mask_apply_method=args.mask_apply_method,
                                                    models_use_deepcopy=False,
                                                    exclusive_dropout=args.exclusive_dropout,
                                                    gradation_coefficients=optimized_lambdas)


    save_instruct_model_path = save_math_model_path = save_code_model_path = save_jp_model_path = None
    if args.merge_instruct:
        save_instruct_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/instruct/{args.dataset_name}/{args.save_model_name}"
    if args.merge_math:
        save_math_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/math/{args.dataset_name}/{args.save_model_name}"
    if args.merge_code:
        save_code_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/code/{args.dataset_name}/{args.save_model_name}"
    if args.merge_jp:
        save_jp_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/jp/{args.dataset_name}/{args.save_model_name}"

    # since the tokenizers of different tasks are different, we need to save them (together with the model) separately
    save_model_paths = [save_instruct_model_path, save_math_model_path, save_code_model_path, save_jp_model_path]
    index = 0
    for save_model_path in save_model_paths:
        if save_model_path==save_instruct_model_path:
            continue
        
        if save_model_path is not None:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            logger.info(f"saving models at {save_model_path}...")
            merged_model.save_pretrained(save_directory=save_model_path)
            tokenizers[index].save_pretrained(save_directory=save_model_path)
            index += 1
            logger.info(f"models are saved")

                
        if args.dataset_name=="gsm8k" and save_model_path==save_math_model_path:
            logger.info(f"evaluating merged model on math task...")
            delete_all_models()
            aggressive_clear_gpu_memory()
            llm = create_llm(finetuned_model_name=save_math_model_path, pretrained_model_name=args.pretrained_model_name,
                            args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                            just_inference=True, save_model_path=None)
            try:
                test_data_path = "math_code_data/gsm8k_test.jsonl"
                test_gsm8k(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                        start_index=args.start_index, end_index=args.end_index, save_model_path=save_model_path, 
                        comp_file_path=args.comp_file_path, model_name=args.save_model_name)
            except Exception as e:
                logger.error(f"gsm8k評価エラー: {str(e)}")
            ""
            try:
                test_data_path = "math_code_data/MATH_test.jsonl"
                test_hendrycks_math(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                                    start_index=args.start_index, end_index=args.end_index, save_model_path=None,
                                    comp_file_path= args.comp_file_path, model_name=args.save_model_name)
            except Exception as e:
                logger.error(f"MATH評価エラー: {str(e)}")
            ""

        elif args.dataset_name=="mbpp" and save_model_path==save_code_model_path:
            logger.info(f"evaluating merged model on code task...")
            delete_all_models()
            aggressive_clear_gpu_memory()
            llm = create_llm(finetuned_model_name=save_code_model_path, pretrained_model_name=args.pretrained_model_name,
                            args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                            just_inference=True, save_model_path=None)
            ""
            try:
                save_gen_results_folder = f"./save_gen_codes_results/{'_'.join(merge_task_names)}/human_eval/{args.save_model_name}"
                os.makedirs(save_gen_results_folder, exist_ok=True)
                test_human_eval(llm=llm, args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                                save_model_path=None, save_gen_results_folder=save_gen_results_folder)
            except Exception as e:
                logger.error(f"human_eval評価エラー: {str(e)}")
            ""
            try:
                save_gen_results_folder = f"./save_gen_codes_results/{'_'.join(merge_task_names)}/mbpp/{args.save_model_name}"
                os.makedirs(save_gen_results_folder, exist_ok=True)
                test_data_path = "math_code_data/mbpp.test.jsonl"
                test_mbpp(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                        start_index=args.start_index, end_index=args.end_index,
                        save_model_path=None, save_gen_results_folder=save_gen_results_folder)
            except Exception as e:
                logger.error(f"mbpp評価エラー: {str(e)}")
            
        
        elif args.dataset_name=="alpaca_eval" and save_model_path==save_instruct_model_path:
            logger.info(f"evaluating merged model on instruct task...")
            delete_all_models()
            aggressive_clear_gpu_memory()
            llm = create_llm(finetuned_model_name=save_instruct_model_path, pretrained_model_name=args.pretrained_model_name,
                            args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                            just_inference=True, save_model_path=None)
            save_gen_results_folder = f"./save_gen_instruct_responses_results/{'_'.join(merge_task_names)}/alpaca_eval/{args.save_model_name}"
            os.makedirs(save_gen_results_folder, exist_ok=True)
            try:
                test_alpaca_eval(llm=llm, finetuned_model_name=save_instruct_model_path,
                                args=args, logger=logger, start_index=args.start_index, end_index=args.end_index,
                                save_model_path=None, save_gen_results_folder=save_gen_results_folder)
            except Exception as e:
                logger.error(f"AlpacaEval評価エラー: {str(e)}")

        elif args.dataset_name=="ja_mgsm" and save_model_path==save_jp_model_path:
            logger.info(f"evaluating merged model on ja_mgsm task...")
            delete_all_models()
            aggressive_clear_gpu_memory()
            llm = create_llm(finetuned_model_name=save_jp_model_path, pretrained_model_name=args.pretrained_model_name,
                            args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                            just_inference=True, save_model_path=None)
            try:
                test_data_path = "juletxara/mgsm"
                test_ja_mgsm(llm=llm, test_data_path=test_data_path, args=args, logger=logger,
                        start_index=args.start_index, end_index=args.end_index, 
                        comp_file_path=args.comp_file_path, model_name=args.save_model_name, drop_rate=args.weight_mask_rate,
                        log_resp_path=args.log_resp_path)
            except Exception as e:
                logger.error(f"ja_mgsm評価エラー: {str(e)}")
                
        if save_model_path is not None:
            shutil.rmtree(save_model_path, ignore_errors=True)

    del merged_model, tokenizers
    torch.cuda.empty_cache()    
    
    for save_model_path in save_model_paths:
        if save_model_path is not None:
            shutil.rmtree(save_model_path, ignore_errors=True)
    logger.info(f"inference of merging method {args.merging_method_name} is completed")
    """

parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--merge_jp1", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_jp2", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_jp", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_bio", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_math1", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_math2", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_math3", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_instruct", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_math", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_code", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merging_method_name", type=str, default="average_merging", help="name of the method to merge models",
                    choices=["average_merging", "task_arithmetic", "mask_merging", "ties_merging"])
parser.add_argument("--scaling_coefficient", type=float, default=1.0, help="scaling coefficient to merge the task vector")
parser.add_argument("--param_value_mask_rate", type=float, default=0.8, help="param_value_mask_rate")
parser.add_argument("--weight_format", type=str, help="the format of weights to be masked", default="delta_weight", choices=["finetuned_weight", "delta_weight"])
parser.add_argument("--weight_mask_rate", type=float, default=0.0, help="weight mask rate")
parser.add_argument("--use_weight_rescale", action="store_true", default=False, help="whether to rescale the weight by 1 / (1 - weight_mask_rate)")
parser.add_argument("--mask_strategy", type=str, help="mask strategy", default="random", choices=["random", "magnitude"])
parser.add_argument("--mask_apply_method", type=str, default="average_merging", help="merging method that the mask strategy applies",
                    choices=["average_merging", "task_arithmetic", "ties_merging"])
parser.add_argument('--start_index', type=int, default=0)
parser.add_argument('--end_index', type=int, default=sys.maxsize)
parser.add_argument("--tensor_parallel_size", type=int, default=1, help="numbers of gpus to use")
parser.add_argument("--comp_file_path", default=None, help="whether to save llm result to compare to others")
parser.add_argument("--log_resp_path", default=None, help="whether to save all response")
parser.add_argument("--exclusive_dropout", action="store_true", default=False, help="exclusive drop")
parser.add_argument("--seed", type=int, default=0, help="seed")
parser.add_argument("--subordinate_mask", action="store_true", default=False, help="subordinate_mask")
parser.add_argument("--single_exclusive_model", action="store_true", default=False, help="single_exclusive_model")
parser.add_argument("--gradation1", type=float, default=None, help="gradation1")
parser.add_argument("--gradation2", type=float, default=None, help="gradation2")
parser.add_argument("--gradation3", type=float, default=None, help="gradation3")
parser.add_argument("--dataset_name", type=str, default=None, help="dataset to be used", choices=["alpaca_eval", "gsm8k", "MATH", "human_eval", "mbpp", "ja_mgsm"])
parser.add_argument("--metagpt", action="store_true", default=False, help="metagpt")
parser.add_argument(
    '--metagpt_strategy',
    type=str,
    choices=[s.value for s in WeightingStrategy],
    default=WeightingStrategy.COSINE_SIMILARITY.value,
    help='Weighting strategy to use'
)
parser.add_argument(
    '--lambda_strategy',
    type=str,
    choices=[s.value for s in WeightingStrategy],
    default=WeightingStrategy.METAGPT.value,
    help='Lambdaの計算戦略を選択'
)
parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for lambda optimization")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for lambda optimization")
parser.add_argument("--num_train_samples", type=int, default=10, help="Number of training samples for lambda optimization")
parser.add_argument("--optimizer_type", type=str, default="spsa", choices=["adam", "sgd", "spsa"], help="Optimizer type for lambda optimization")
parser.add_argument("--num_steps", type=int, default=5, help="Number of steps for lambda optimization")
parser.add_argument("--initial_lambda_filepath", type=str, default=None, help="initial lambda filepath")
parser.add_argument("--optimized_lambda_filepath", type=str, default=None, help="optimized lambda filepath")
parser.add_argument("--run_name", type=str, default=None, help="run name")


try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit()

if __name__ == "__main__":
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    assert sum([args.merge_math1, args.merge_math2, args.merge_math3, args.merge_jp1, args.merge_jp2, args.merge_bio, args.merge_instruct, args.merge_math, args.merge_code, args.merge_jp]) >= 2, "should merge two tasks at least!"
    finetuned_model_names = []
    merge_task_names = []
    for merge_flag, task_name in zip([args.merge_math1, args.merge_math2, args.merge_math3, args.merge_jp1, args.merge_jp2, args.merge_bio, args.merge_instruct, args.merge_math, args.merge_code, args.merge_jp], ["math1", "math2", "math3", "jp1", "jp2", "bio", "instruct", "math", "code", "jp"]):
        if merge_flag:
            finetuned_model_names.append(task_model_mapping_dict[task_name])
            merge_task_names.append(task_name)
    args.merge_task_names = merge_task_names 

    pretrained_model_names = [finetuned_model_backbone_mapping_dict[finetuned_model_name] for finetuned_model_name in finetuned_model_names]
    assert len(set(pretrained_model_names)) == 1, "the backbone of all the finetuned models should be the same!"
    args.pretrained_model_name = pretrained_model_names[0]
    args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    if args.merging_method_name == "average_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "task_arithmetic" or args.merging_method_name == "ties_merging":
        #args.save_model_name = f"{args.merging_method_name}_gr1_{args.gradation1}_gr2_{args.gradation2}_gr3_{args.gradation3}"
        args.save_model_name = f"{args.run_name}"
    else:
        assert args.merging_method_name == "mask_merging"
        if args.mask_apply_method == "average_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"
        else:
            assert args.mask_apply_method == "task_arithmetic" or args.mask_apply_method == "ties_merging"
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}"
        weight_mask_rates = [str(weight_mask_rate) for weight_mask_rate in args.weight_mask_rates]
        if args.exclusive_dropout:
            args.save_model_name = f"{args.merging_method_name}/{mask_apply_method_name}/exclusive_mask_{'_'.join(weight_mask_rates)}_rescale_{args.use_weight_rescale}_grad1_{args.gradation1}_grad2_{args.gradation2}_grad3_{args.gradation3}"
        else:
            args.save_model_name = f"{args.merging_method_name}/{mask_apply_method_name}/mask_{'_'.join(weight_mask_rates)}_rescale_{args.use_weight_rescale}_grad1_{args.gradation1}_grad2_{args.gradation2}_grad3_{args.gradation3}"

    save_merge_log_path = f"./save_merge_llm_logs/{'_'.join(merge_task_names)}/{args.save_model_name}"
    args.model_name_in_comp_file = f"{'_'.join(merge_task_names)}_{args.save_model_name}"
    os.makedirs(save_merge_log_path, exist_ok=True)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"{save_merge_log_path}/{str(time.time())}.log")
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    run_start_time = time.time()
    logger.info(f"********** Run starts. **********")

    models_to_merge = []
    finetuned_tokenizers = []
    merging_method = MergingMethod(merging_method_name=args.merging_method_name)
    for finetuned_model_name in finetuned_model_names:
        finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), device_map="cpu", torch_dtype=torch.bfloat16, load_in_4bit=True)
        finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name),)
        finetuned_model.gradient_checkpointing_enable()
        models_to_merge.append(finetuned_model)
        finetuned_tokenizers.append(finetuned_tokenizer)

    get_merge_performance(args=args, finetuned_model_names=finetuned_model_names, merge_task_names=merge_task_names, models_to_merge=models_to_merge,
                          trainers=[None for _ in range(len(finetuned_model_names))], logger=logger, merging_method=merging_method, tokenizers=finetuned_tokenizers)

    sys.exit()