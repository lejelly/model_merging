import torch
import argparse
import sys
import os
import shutil
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_merging_methods.merging_methods import MergingMethod
from utils.utils import set_random_seed, smart_tokenizer_and_embedding_resize
from inference_llms_instruct_math_code import create_llm, test_alpaca_eval, test_gsm8k, test_hendrycks_math, test_human_eval, test_mbpp, test_ja_mgsm
from utils.load_config import cache_dir

task_model_mapping_dict = {
    "jp1": "augmxnt/shisa-gamma-7b-v1",
    "jp2": "tokyotech-llm/Swallow-MS-7b-v0.1",
    "math1": "WizardLMTeam/WizardMath-7B-V1.1",
    "math2": "GAIR/Abel-7B-002",
    "math3": "upaya07/Arithmo2-Mistral-7B",
    "bio": "BioMistral/BioMistral-7B",
}
finetuned_model_backbone_mapping_dict = {
    "WizardLMTeam/WizardMath-7B-V1.1": "mistralai/Mistral-7B-v0.1",
    "augmxnt/shisa-gamma-7b-v1": "mistralai/Mistral-7B-v0.1",
    "GAIR/Abel-7B-002": "mistralai/Mistral-7B-v0.1",
    "tokyotech-llm/Swallow-MS-7b-v0.1": "mistralai/Mistral-7B-v0.1",
    "BioMistral/BioMistral-7B": "mistralai/Mistral-7B-v0.1",
    "upaya07/Arithmo2-Mistral-7B": "mistralai/Mistral-7B-v0.1",
}

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
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name), device_map="cpu", torch_dtype=torch.bfloat16)
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, args.pretrained_model_name))
    except:
        pretrained_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir, device_map="cpu", torch_dtype=torch.bfloat16)
        pretrained_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.pretrained_model_name, cache_dir=cache_dir)

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
    set_random_seed(seed=0)
    merged_model = pretrained_model
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
                                                   exclusive_dropout=args.exclusive_dropout)

    save_jp1_model_path = save_jp2_model_path = save_bio_model_path = save_math1_model_path = save_math2_model_path = save_math3_model_path = None
    if args.merge_jp1:
        save_jp1_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/jp1/{args.save_model_name}"
    if args.merge_jp2:
        save_jp2_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/jp2/{args.save_model_name}"
    if args.merge_bio:
        save_bio_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/bio/{args.save_model_name}"
    if args.merge_math1:
        save_math1_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/math1/{args.save_model_name}"
    if args.merge_math2:
        save_math2_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/math2/{args.save_model_name}"
    if args.merge_math3:
        save_math3_model_path = f"./save_merge_models/{'_'.join(merge_task_names)}/math3/{args.save_model_name}"

    # since the tokenizers of different tasks are different, we need to save them (together with the model) separately
    save_model_paths = [save_jp1_model_path, save_jp2_model_path, save_bio_model_path, save_math1_model_path, save_math2_model_path, save_math3_model_path]
    index = 0
    for save_model_path in save_model_paths:
        if save_model_path is not None:
            logger.info(f"saving models at {save_model_path}...")
            merged_model.save_pretrained(save_directory=save_model_path)
            tokenizers[index].save_pretrained(save_directory=save_model_path)
            index += 1
    logger.info(f"models are saved")
    del merged_model, tokenizers
    
    if save_jp1_model_path is not None:
        logger.info(f"evaluating merged model on math task...")
        llm = create_llm(finetuned_model_name=save_jp1_model_path, pretrained_model_name=args.pretrained_model_name,
                            args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                            just_inference=True, save_model_path=None)
        args.test_data_path = "juletxara/mgsm"
        test_ja_mgsm(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger,
                      start_index=args.start_index, end_index=args.end_index, 
                      comp_file_path=args.comp_file_path, model_name=args.model_name_in_comp_file, drop_rate=args.weight_mask_rates,
                      log_resp_path=args.log_resp_path)
    elif save_math1_model_path is not None:
        logger.info(f"evaluating merged model on math task...")
        llm = create_llm(finetuned_model_name=save_math1_model_path, pretrained_model_name=args.pretrained_model_name,
                            args=args, logger=logger, tensor_parallel_size=args.tensor_parallel_size,
                            just_inference=True, save_model_path=None)
        args.test_data_path = "juletxara/mgsm"
        test_ja_mgsm(llm=llm, test_data_path=args.test_data_path, args=args, logger=logger,
                      start_index=args.start_index, end_index=args.end_index, 
                      comp_file_path=args.comp_file_path, model_name=args.model_name_in_comp_file, drop_rate=args.weight_mask_rates,
                      log_resp_path=args.log_resp_path)

    for save_model_path in save_model_paths:
        if save_model_path is not None:
            shutil.rmtree(save_model_path, ignore_errors=True)
    logger.info(f"inference of merging method {args.merging_method_name} is completed")


parser = argparse.ArgumentParser("Interface for merging LLMs")
parser.add_argument("--merge_jp1", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_jp2", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_bio", action="store_true", default=False, help="whether to merge instruct model")
parser.add_argument("--merge_math1", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_math2", action="store_true", default=False, help="whether to merge math model")
parser.add_argument("--merge_math3", action="store_true", default=False, help="whether to merge math model")
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

    assert sum([args.merge_jp1, args.merge_jp2, args.merge_bio, args.merge_math1, args.merge_math2, args.merge_math3]) >= 2, "should merge two tasks at least!"
    finetuned_model_names = []
    merge_task_names = []
    for merge_flag, task_name in zip([args.merge_jp1, args.merge_jp2, args.merge_bio, args.merge_math1, args.merge_math2, args.merge_math3], ["jp1", "jp2", "bio", "math1", "math2", "math3"]):
        if merge_flag:
            finetuned_model_names.append(task_model_mapping_dict[task_name])
            merge_task_names.append(task_name)

    pretrained_model_names = [finetuned_model_backbone_mapping_dict[finetuned_model_name] for finetuned_model_name in finetuned_model_names]
    assert len(set(pretrained_model_names)) == 1, "the backbone of all the finetuned models should be the same!"
    args.pretrained_model_name = pretrained_model_names[0]
    args.weight_mask_rates = [args.weight_mask_rate for _ in range(len(finetuned_model_names))]

    if args.merging_method_name == "average_merging":
        args.save_model_name = f"{args.merging_method_name}"
    elif args.merging_method_name == "task_arithmetic" or args.merging_method_name == "ties_merging":
        args.save_model_name = f"{args.merging_method_name}_scaling_coefficient_{args.scaling_coefficient}"
    else:
        assert args.merging_method_name == "mask_merging"
        if args.mask_apply_method == "average_merging":
            mask_apply_method_name = f"{args.mask_apply_method}"
        else:
            assert args.mask_apply_method == "task_arithmetic" or args.mask_apply_method == "ties_merging"
            mask_apply_method_name = f"{args.mask_apply_method}_scaling_coefficient_{args.scaling_coefficient}"
        weight_mask_rates = [str(weight_mask_rate) for weight_mask_rate in args.weight_mask_rates]
        args.save_model_name = f"{args.merging_method_name}/{mask_apply_method_name}/mask_{'_'.join(weight_mask_rates)}_rescale_{args.use_weight_rescale}"

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
        finetuned_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name), device_map="cpu", torch_dtype=torch.bfloat16)
        finetuned_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=os.path.join(cache_dir, finetuned_model_name),)
        models_to_merge.append(finetuned_model)
        finetuned_tokenizers.append(finetuned_tokenizer)

    get_merge_performance(args=args, finetuned_model_names=finetuned_model_names, merge_task_names=merge_task_names, models_to_merge=models_to_merge,
                          trainers=[None for _ in range(len(finetuned_model_names))], logger=logger, merging_method=merging_method, tokenizers=finetuned_tokenizers)

    sys.exit()
