# DARE: Drop And RE-scale
- Paper: [Language Models are Super Mario: Absorbing Abilities from Homologous Models as a Free Lunch](https://arxiv.org/abs/2311.03099)

- code: [official repogitory](https://github.com/yule-BUAA/MergeLM?tab=readme-ov-file) 


# Model(for now)
You can add models in merge_llms_instruct_math_code.py
### BASE MODEL
- [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
### FINE-TUNED MODEL
- [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [lightblue/suzume-llama-3-8B-japanese](https://huggingface.co/lightblue/suzume-llama-3-8B-japanese)

# Quick start
```
huggingface-cli login
```
## Single model inference
Fill in [MODEL NAME] .

### w/o DARE (drop rate 0.0)
```
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name [MODEL NAME] --tensor_parallel_size 1 --weight_mask_rate 0.0
```

### Drop Only (drop rate 0.9)
```
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name [MODEL NAME] --tensor_parallel_size 1 --weight_mask_rate 0.9
```

### Magnitude-Based Pruning (drop rate 0.9)
```
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name [MODEL NAME] --tensor_parallel_size 1 --weight_mask_rate 0.9 --mask_strategy magnitude
```

### Masking Fine-Tuned Parameters (drop rate 0.9) 
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name [MODEL NAME] --tensor_parallel_size 1 --weight_mask_rate 0.9 --use_weight_rescale --weight_format finetuned_weight

### DARE (drop rate 0.9 and Re-scale)
```
python inference_llms_instruct_math_code.py --dataset_name gsm8k --finetuned_model_name [MODEL NAME] --tensor_parallel_size 1 --weight_mask_rate 0.9 --use_weight_rescale
```

## Merging model inference
Fill in [MODEL NAME1] and [MODEL NAME2] .

### Avg Merging
```
python merge_llms_instruct_math_code.py --[MODEL NAME1] --[MODEL NAME2] --merging_method_name average_merging --tensor_parallel_size 1
```

### Task Arithmetic
```
python merge_llms_instruct_math_code.py --[MODEL NAME1] --[MODEL NAME2] --merging_method_name task_arithmetic --scaling_coefficient 1.0 --tensor_parallel_size 1
```

### AvgMerging and DARE (drop rate 0.9 and Re-scale)
```
python merge_llms_instruct_math_code.py --[MODEL NAME1] --[MODEL NAME2] --merging_method_name mask_merging --use_weight_rescale --weight_mask_rate 0.9 --mask_apply_method average_merging --tensor_parallel_size 1
```