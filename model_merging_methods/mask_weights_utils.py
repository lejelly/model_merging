from tqdm import tqdm
import torch
import torch.nn as nn

from utils.utils import get_param_names_to_merge
from model_merging_methods.task_vector import TaskVector, NewTaskVector
from memory_profiler import profile
from typing import Dict, Iterator, Tuple
import copy

def mask_input_with_mask_rate(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        masked_input_tensor = input_tensor * (1 - mask)
    else:
        assert mask_strategy == "magnitude", f"wrong setting for mask_strategy {mask_strategy}!"
        original_shape = input_tensor.shape
        input_tensor = input_tensor.flatten()
        num_mask_params = int(len(input_tensor) * mask_rate)
        # Tensor, shape (1, ), find the num_mask_params-th smallest magnitude element of all the parameters in the model
        kth_values, _ = input_tensor.abs().kthvalue(k=num_mask_params, dim=0, keepdim=True)
        # Tensor, shape (num_total_params, ), where True is for parameters that we want to perform mask
        mask = input_tensor.abs() <= kth_values
        masked_input_tensor = input_tensor * (~mask)
        masked_input_tensor = masked_input_tensor.reshape(original_shape)
    if use_rescale and mask_rate != 1.0:
        masked_input_tensor = torch.div(input=masked_input_tensor, other=1 - mask_rate)
    return masked_input_tensor

@profile
def mask_model_weights(finetuned_model: nn.Module, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    """
    mask model weights
    :param finetuned_model: nn.Module, the finetuned model
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    
    # get weights that need to be masked
    if weight_format == "finetuned_weight":
        param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
        # exclude parameter whose name matches element in exclude_param_names_regex
        param_names_to_merge = get_param_names_to_merge(input_param_names=list(param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = {param_name: param_dict[param_name] for param_name in param_names_to_merge}
    else:
        assert weight_format == "delta_weight", f"wrong setting for weight_format {weight_format}!"
        task_vector = TaskVector(pretrained_model=pretrained_model, finetuned_model=finetuned_model, exclude_param_names_regex=exclude_param_names_regex)
        model_param_dict = task_vector.task_vector_param_dict

    with torch.no_grad():
        masked_params = {}
        for param_name, param_value in model_param_dict.items():
            masked_param = mask_input_with_mask_rate(
                input_tensor=param_value, 
                mask_rate=weight_mask_rate,
                use_rescale=use_weight_rescale, 
                mask_strategy=mask_strategy
            )
            masked_params[param_name] = masked_param
            yield param_name, masked_param
        
        if weight_format == "delta_weight":
            new_task_vector = TaskVector(task_vector_param_dict=masked_params)
            # combine with parameters of the merged model based on scaling coefficient
            yield from new_task_vector.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0)


def simple_mask_input(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        mask = torch.bernoulli(torch.full_like(input=input_tensor, fill_value=mask_rate)).to(input_tensor.device)
        input_tensor.mul_(1 - mask)
    if use_rescale and mask_rate != 1.0:
        input_tensor.div_(1 - mask_rate)
    return input_tensor, mask

def exclusive_mask_input(input_tensor: torch.Tensor, mask_rate: float, use_rescale: bool, mask_strategy: str, mask: torch.Tensor):
    """
    mask the input with mask rate
    :param input_tensor: Tensor, input tensor
    :param mask_rate: float, mask rate
    :param use_rescale: boolean, whether to rescale the input by 1 / (1 - mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return:
    """
    assert 0.0 <= mask_rate <= 1.0, f"wrong range of mask_rate {mask_rate}, should be [0.0, 1.0]!"
    if mask_strategy == "random":
        #exclusive_mask = 1 - mask
        #masked_input_tensor = input_tensor * (1 - exclusive_mask) # 元実装が1つ目のモデルに 1-mask をかけているので、exclusive_mask(=mask)はちゃんと排他的になっている
        
        mask.neg_() # in-place操作で mask の各要素を負にする  [0, 1, 0, 1]->[0, -1, 0, -1]
        mask.add_(1) # mask の各要素に1を加えます [0, -1, 0, -1]->[1, 0, 1, 0]
        input_tensor.mul_(mask)
        del mask
    if use_rescale and mask_rate != 1.0:
        input_tensor.div_(1 - mask_rate)
    return input_tensor

def mask_model_weights_exclusive(finetuned_models: list, pretrained_model: nn.Module, exclude_param_names_regex: list, weight_format: str,
                       weight_mask_rate: float, use_weight_rescale: bool, mask_strategy: str):
    """
    mask model weights
    :param finetuned_models: list of nn.Module, the finetuned models
    :param pretrained_model: nn.Module, the pretrained model
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :param weight_format: str, the format of weights to be masked, can be "finetuned_weight" and "delta_weight"
    :param weight_mask_rate: float, weight mask rate
    :param use_weight_rescale: boolean, whether to rescale the weight by 1 / (1 - weight_mask_rate)
    :param mask_strategy: str, mask strategy, can be "random" and "magnitude"
    :return: masked_param_dicts: list
    """

    assert weight_format == "delta_weight", f"wrong setting for weight_format {weight_format}!"
    assert len(finetuned_models) == 2, f"This function currently supports exactly 2 models, but {len(finetuned_models)} were provided."
    
    task_vectors = [TaskVector(pretrained_model=pretrained_model, finetuned_model=fm, exclude_param_names_regex=exclude_param_names_regex) for fm in finetuned_models]
    model_param_dicts = [tv.task_vector_param_dict for tv in task_vectors]
    
    masked_param_dicts = [{}, {}]

    with torch.no_grad():
        for (param_name1, param_value1),(param_name2, param_value2) in zip(model_param_dicts[0].items(), model_param_dicts[1].items()):
            assert param_name1 == param_name2, f"Wrong params: param_name1 {param_name1}, param_name2 {param_name2}!"
            masked_param_dicts[0][param_name1], mask = simple_mask_input(input_tensor=param_value1, mask_rate=weight_mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy)
            masked_param_dicts[1][param_name2] = exclusive_mask_input(input_tensor=param_value2, mask_rate=weight_mask_rate,
                                                                      use_rescale=use_weight_rescale, mask_strategy=mask_strategy, mask=mask)

        new_task_vectors = [TaskVector(task_vector_param_dict=mpd) for mpd in masked_param_dicts]
        masked_param_dicts = [ntv.combine_with_pretrained_model(pretrained_model=pretrained_model, scaling_coefficient=1.0) for ntv in new_task_vectors]

    return masked_param_dicts