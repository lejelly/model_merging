# DARE
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
### Example of merging Llama-3-8B-Instruct and llama-3-8B-japanese-gguf with Average Merging and DARE (drop rate 0.2):
```
python merge_llms_instruct_math_code.py --merge_instruct --merge_japanese --merging_method_name mask_merging --use_weight_rescale --weight_mask_rate 0.2 --mask_apply_method average_merging --tensor_parallel_size 1
```