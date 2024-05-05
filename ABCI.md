# Quik Start
We don't use Singurality.
## Set Up
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
Deactivating a virtual environment
```
deactivate
```
Start interactive jobs
```
qrsh -g gcd50654 -l rt_AG.small=1 -l h_rt=1:00:00
```

