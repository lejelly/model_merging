# Quik Start
We don't use Singurality.
## Set Up for ABCI
Creation of a virtual environment: work
```
cd /home/acf15429bz/model_merging/model_merging
```
```
module load python/3.10/3.10.14
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
Deactivating a virtual environment
```
deactivate
```

## Job Execution
### Start interactive jobs
```
qrsh -g gcb50389 -l rt_AG.small=1 -l h_rt=1:00:00
```
environment setup
```
module load python/3.10/3.10.14 cuda/12.1/12.1.1 cudnn/8.9/8.9.7 
```
```
cd /home/acf15429bz/model_merging/model_merging
```
```
source work/bin/activate
```
```
huggingface-cli login
```
```
export HF_HOME=/scratch/acf15429bz/model_merge/pretrained_models
export CUDA_HOME=/apps/cuda/12.1.1
```
```
python xxxxx.py --options
```

### Start Batch jobs
```
cd /home/acf15429bz/model_merging/model_merging
```
```
qsub -g gcb50389 scripts/xxxxx.sh --options
```