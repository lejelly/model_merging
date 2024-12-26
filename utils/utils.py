import re
import os
from typing import Dict
import random
import numpy as np
import torch
import torch.nn as nn
import transformers
from transformers import Trainer, TrainerState
from fasttext.FastText import _FastText
import gc
import json

def aggressive_clear_gpu_memory():
    # 現在のメモリ状態を表示
    #print("Before cleanup:")
    #for i in range(torch.cuda.device_count()):
    #    print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
    #    print(f"GPU {i} memory cached: {torch.cuda.memory_cached(i) / 1024**2:.2f} MB")
    
    try:
        # 1. すべての変数を削除
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
                elif hasattr(obj, 'cache_dir'):
                    del obj
                elif hasattr(obj, 'state_dict'):
                    del obj
            except Exception:
                pass

        # 2. キューに入っているすべてのGPUコマンドを同期
        torch.cuda.synchronize()

        # 3. 各GPUのストリームを同期
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(f'cuda:{i}'):
                    torch.cuda.synchronize()
                    current_stream = torch.cuda.current_stream()
                    current_stream.synchronize()
                    
        # 4. CUDAキャッシュを強制的にクリア
        torch.cuda.empty_cache()
        
        # 5. IPCキャッシュをクリア
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()
        
        # 6. ガベージコレクションを複数回実行
        for _ in range(3):
            gc.collect()
        
        # 7. 環境変数を使用してCUDAキャッシュを制限
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
        
        # 8. メモリの断片化を防ぐために小さなテンソルを作成して削除
        dummy = torch.cuda.FloatTensor(1).fill_(0)
        del dummy
        
    except Exception as e:
        print(f"Error during memory cleanup: {e}")
    
    #finally:
        # 最終的なメモリ状態を表示
        #print("\nAfter cleanup:")
        #for i in range(torch.cuda.device_count()):
        #    print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        #    print(f"GPU {i} memory cached: {torch.cuda.memory_cached(i) / 1024**2:.2f} MB")

# モデルを含むすべての変数を削除する関数
def delete_all_models():
    for var in list(globals()):
        try:
            if 'model' in var.lower() or 'llm' in var.lower():
                del globals()[var]
        except Exception:
            pass

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_state_and_model_for_hf_trainer(trainer: Trainer):
    """
    save the state and model for trainer
    :param trainer: transformers.Trainer to be saved
    :return:
    """
    # save trainer state at trainer.args.output_dir path
    trainer.save_state()
    # save model at output_dir
    if trainer.args.should_save:
        # convert state_dict to cpu
        cpu_state_dict = {key: value.cpu() for key, value in trainer.model.state_dict().items()}
        trainer._save(trainer.args.output_dir, state_dict=cpu_state_dict)


def load_state_and_model_for_hf_trainer(model: nn.Module, load_model_dir: str, map_location: str = None):
    """
    load the state and model for trainer
    :param model: nn.Module, the model to be loaded
    :param load_model_dir: str, the path where the state and model to be loaded
    :param map_location: str, how to remap the storage locations
    :return:
    """
    # load model and trainer state from load_model_dir
    model.load_state_dict(torch.load(os.path.join(load_model_dir, "pytorch_model.bin"), map_location=map_location))
    # model = model.from_pretrained(load_model_dir)
    trainer_state = TrainerState.load_from_json(os.path.join(load_model_dir, "trainer_state.json"))
    return model, trainer_state


def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge


def get_modules_to_merge(model: nn.Module, include_module_types: list):
    """
    get the model modules that need to be merged, whose type is in include_module_types
    :param model: nn.Module, input model
    :param include_module_types: list, module types that want to include
    :return:
    """
    modules_to_merge = {}
    for module_name, module in model.named_modules():
        is_valid_type = not include_module_types or any([isinstance(module, include_module_type) for include_module_type in include_module_types])
        if is_valid_type:
            modules_to_merge[module_name] = module
    return modules_to_merge


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    #assert tokenizer.vocab_size == 128000
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    if num_new_tokens > 0:
        model.resize_token_embeddings(tokenizer.vocab_size + num_new_tokens)

        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

class LanguageDetector:
    def __init__(self):
        model_path = os.environ.get("LID176FTZ_PATH", "lid.176.ftz")
        self.model = _FastText(model_path)

    def __call__(self, text: str) -> dict:
        return dict(zip(*self.model.predict(text.replace("\n", ""), k=-1)))


def copy_params_to_model(params: dict, model: nn.Module):
    """
    copy parameters in "params" to the model
    :param params: dict, dictionary of parameters
    :param model: nn.Module, model that needs to copy parameters
    :return:
    """
    for param_name, param_value in model.named_parameters():
        if param_name in params:
            param_value.data.copy_(params[param_name])

def stream_jsonl(filename: str):
    """JSONLファイルを1行ずつ読み込むジェネレータ関数"""
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():  # 空行をスキップ
                yield json.loads(line)

def write_jsonl(filename: str, data_list):
    """データをJSONL形式でファイルに書き込む"""
    with open(filename, 'w') as f:
        for item in data_list:
            f.write(json.dumps(item) + '\n')
