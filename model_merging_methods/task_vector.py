import torch
import torch.nn as nn

from utils.utils import get_param_names_to_merge
from memory_profiler import profile
from typing import Dict, Iterator, Tuple

class TaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    # 元(DARE MARIO)のコード：メモリ使用量が大きいためコメントアウト
                    #self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
                    delta = finetuned_param_dict[param_name]
                    delta.sub_(pretrained_param_dict[param_name])  # 差分を直接更新
                    self.task_vector_param_dict[param_name] = delta  # 更新したテンソルを格納
                    del pretrained_param_dict[param_name], finetuned_param_dict[param_name]  # 使わないテンソルを削除してメモリ解放

    def __add__(self, other):
        """
        add task vector
        :param other: TaskVector to add, at right side
        :return:
        """
        assert isinstance(other, TaskVector), "addition of TaskVector can only be done with another TaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return TaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: TaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        
        with torch.no_grad():
            for param_name, pretrained_param in pretrained_model.named_parameters():
                if param_name in self.task_vector_param_dict:
                    # 新しいテンソルを作成するが、中間結果は保持しない
                    yield param_name, pretrained_param + scaling_coefficient * self.task_vector_param_dict[param_name]
                else:
                    # タスクベクトルに存在しないパラメータはそのまま使用
                    yield param_name, pretrained_param
        
        """
        
        #方法1：メモリ使用量を削減するため、直接更新する(gpu複数枚)
        pretrained_param_iter = pretrained_model.named_parameters()
        with torch.no_grad():
            merged_params = {}
            for param_name, pretrained_param in pretrained_param_iter:
                if param_name in self.task_vector_param_dict:
                    # 新しいテンソルを作成するが、中間結果は保持しない
                    merged_params[param_name] = pretrained_param + scaling_coefficient * self.task_vector_param_dict[param_name]
                else:
                    # タスクベクトルに存在しないパラメータはそのまま使用
                    merged_params[param_name] = pretrained_param

        return merged_params
        """

        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        #方法2：メモリ使用量を削減するため、直接更新する(gpu複数枚)
        # 元（DARE MARIO）のコード：メモリ使用量が大きいためコメントアウト
        with torch.no_grad():
            merged_params = {}
            for param_name in self.task_vector_param_dict:
                merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]
        
        return merged_params
        """
        """
        #方法３：メモリ使用量を削減するため、直接更新する(gpu1枚)
        # メモリ使用量を削減するため、直接更新する
        with torch.no_grad():
            for param_name, task_vector_param in self.task_vector_param_dict.items():
                # インプレースで計算
                pretrained_param_dict[param_name].add_(scaling_coefficient * task_vector_param)
                
        return pretrained_param_dict  # 直接更新された辞書を返す
        """
        

class NewTaskVector:
    def __init__(self, pretrained_model: nn.Module = None, finetuned_model: nn.Module = None, exclude_param_names_regex: list = None, task_vector_param_dict: dict = None):
        """
        Task vector. Initialize the task vector from a pretrained model and a finetuned model, or
        directly passing the task_vector_param_dict dictionary.
        :param pretrained_model: nn.Module, pretrained model
        :param finetuned_model: nn.Module, finetuned model
        :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
        :param task_vector_param_dict: dict, task vector to initialize self.task_vector_param_dict
        """
        if task_vector_param_dict is not None:
            self.task_vector_param_dict = task_vector_param_dict
        else:
            self.task_vector_param_dict = {}
            pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}
            finetuned_param_dict = {param_name: param_value for param_name, param_value in finetuned_model.named_parameters()}
            param_names_to_merge = get_param_names_to_merge(input_param_names=list(pretrained_param_dict.keys()), exclude_param_names_regex=exclude_param_names_regex)
            with torch.no_grad():
                for param_name in param_names_to_merge:
                    # 元(DARE MARIO)のコード：メモリ使用量が大きいためコメントアウト
                    #self.task_vector_param_dict[param_name] = finetuned_param_dict[param_name] - pretrained_param_dict[param_name]
                    delta = finetuned_param_dict[param_name]
                    delta.sub_(pretrained_param_dict[param_name])  # 差分を直接更新
                    self.task_vector_param_dict[param_name] = delta  # 更新したテンソルを格納
                    del delta, pretrained_param_dict[param_name], finetuned_param_dict[param_name]  # 使わないテンソルを削除してメモリ解放
                    torch.cuda.empty_cache()

    def __imul__(self, scalar):
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                self.task_vector_param_dict[param_name] *= scalar
        return self
    
    def __mul__(self, scalar):
        """
        Multiply the task vector by a scalar
        :param scalar: float, scalar to multiply
        :return: NewTaskVector
        """
        assert isinstance(scalar, (int, float)), "Multiplication of NewTaskVector can only be done with a scalar (int or float)!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] * scalar
        return NewTaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __rmul__(self, scalar):
        """
        Right multiplication (scalar * self)
        :param scalar: float, scalar to multiply
        :return: NewTaskVector
        """
        return self.__mul__(scalar)

    def __iadd__(self, other):
        assert isinstance(other, NewTaskVector), "Addition of NewTaskVector can only be done with another NewTaskVector!"
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                self.task_vector_param_dict[param_name] += other.task_vector_param_dict[param_name]
        return self

    def __add__(self, other):
        """
        Add task vector
        :param other: NewTaskVector to add, at right side
        :return: NewTaskVector
        """
        if isinstance(other, (int, float)) and other == 0:
            return self
        assert isinstance(other, NewTaskVector), "Addition of NewTaskVector can only be done with another NewTaskVector!"
        new_task_vector_param_dict = {}
        with torch.no_grad():
            for param_name in self.task_vector_param_dict:
                assert param_name in other.task_vector_param_dict.keys(), f"param_name {param_name} is not contained in both task vectors!"
                new_task_vector_param_dict[param_name] = self.task_vector_param_dict[param_name] + other.task_vector_param_dict[param_name]
        return NewTaskVector(task_vector_param_dict=new_task_vector_param_dict)

    def __radd__(self, other):
        """
        other + self = self + other
        :param other: NewTaskVector to add, at left side
        :return:
        """
        return self.__add__(other)

    def combine_with_pretrained_model(self, pretrained_model: nn.Module, scaling_coefficient: float = 1.0):
        """
        combine the task vector with pretrained model
        :param pretrained_model: nn.Module, pretrained model
        :param scaling_coefficient: float, scaling coefficient to merge the task vector
        :return:
        """
        pretrained_param_dict = {param_name: param_value for param_name, param_value in pretrained_model.named_parameters()}

        # 元（DARE MARIO）のコード：メモリ使用量が大きいためコメントアウト
        #with torch.no_grad():
        #    merged_params = {}
        #    for param_name in self.task_vector_param_dict:
        #        merged_params[param_name] = pretrained_param_dict[param_name] + scaling_coefficient * self.task_vector_param_dict[param_name]
        #
        #return merged_params
    
        # メモリ使用量を削減するため、直接更新する
        task_vector_param_copy = self.task_vector_param_dict.copy()
        with torch.no_grad():
            for param_name, task_vector_param in task_vector_param_copy.items():
                # インプレースで計算
                pretrained_param_dict[param_name].add_(scaling_coefficient * task_vector_param)
                # 元の辞書から削除
                del self.task_vector_param_dict[param_name]
                torch.cuda.empty_cache()  # 必要であればGPUメモリをクリア
        return pretrained_param_dict  # 直接更新された辞書を返す
    
