# Quik Start
We don't use Singurality.
## Set Up for ABCI
Creation of a virtual environment: work
```
cd /home/acf15429bz/model_merging/model_merging
```
```
module load python/3.11/3.11.9
```
```
python3 -m venv work
```
Activating a virtual environment: work
```
source work/bin/activate
```
Installing numpy to a virtual environment
```
pip3 install -r requirements.txt
```
huggingface login
```
huggingface-cli login
```
Environment variables
```
export HF_HOME=/scratch/acf15429bz/model_merge/pretrained_models
```
Deactivating a virtual environment
```
deactivate
```

Change huggingface download destination '~' into global scratch area at [line 97](https://github.com/huggingface/huggingface_hub/blob/5ff2d150d121d04799b78bc08f2343c21b8f07a9/src/huggingface_hub/constants.py#L97)
- Because ABCI home area has only 200GB capacity
- Global scratch area has 10TiB
 
line 97 in /PATH_TO_[YOUR_VATUAL_ENV_NAME]/lib/python3.11/site-packages/huggingface_hub/constants.py 
```
97: #default_home = os.path.join(os.path.expanduser("~"), ".cache")
98: default_home = os.path.join(os.path.expanduser('/scratch/[YOUR_ABCI_ACCOUNT]/model_merge'), ".cache")
```

## Job Execution
Start interactive jobs
```
qrsh -g gcb50389 -l rt_AG.small=1 -l h_rt=1:00:00
```
Start Batch jobs
```
qsub -g gcb50389 scripts/dare.sh
```