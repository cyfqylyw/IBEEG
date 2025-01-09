# IBEEG

Code implementation for **IBEEG**.


<!-- ## Abstract

TODO -->


## Quick Start

Quick start to understand input and output of each model.

TODO



## Requirements

```
conda create -n VIBEEG python=3.9
conda activate VIBEEG
pip install -r requirements.txt
```

## Dataset Preparation and Folder Structure

Download the datasets:
- [TUAB & TUEV](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/), 
- [Crowdsourced](https://osf.io/9bvgh/), 
- [Simultaneous Task EEG Workload (STEW)](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset), 
- [DREAMER](https://zenodo.org/records/546113), 
- [SEED-V Dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html) 

Put them in the folder:
```
IBEEG/
    |_ *.py
    |_ README.md
    |_ requirements.txt
    |_ datasets/
        |_ preprocess_tuab.py
        |_ preprocess_tuev.py
        |_ processed/
        |_ raw/
            |_ DREAMER/
                |_ DREAMER.mat
            |_ STEW Dataset/
                |_ *.txt
            |_ SleepEDF/
                |_ sleep-edf-database-expanded-1.0.0/
                    |_ sleep-telemetry/
            |_ ISRUC-SLEEP/
                |_ Subgroup_1/
                    |_ 1/
                        |_ 1.rec
                        |_ 1.txt
                |_ Subgroup_2/
                |_ Subgroup_3/
            |_ SEED-V/
                |_ EEG_raw/
            |_ Crowsourced/
                |_ Raw Data/
            |_ TUAB/
                |_ edf/
                    |_ train/
                    |_ eval/
            |_ TUEV/
                |_ edf/
                    |_ train/
                    |_ eval/
```

Preprocess for TUAB and TUEV datasets.
```
cd datasets/
python preprocess_tuab.py
python preprocess_tuev.py
```

## Usage

For EEG2Rep Dataset:
```
nohup python -u main.py --dataset dreamer --overlap 0 --epoch 100 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:0 > output_dreamer4.log 2>&1 &
<!-- acc: 0.6845 -->

nohup python -u main.py --dataset stew --overlap 0 --epoch 100 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:3 > output_stew.log 2>&1 &
<!-- acc: 0.8000 -->
```

For Sleep Dataset:
```
nohup python -u main.py --dataset isruc --epoch 50 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --batch_size 32 --device cuda:5 > output_isruc1.log 2>&1 &
nohup python -u main.py --dataset isruc --epoch 100 --lr 1e-4 --alpha 1e-4 --beta 1e-4 --batch_size 32 --device cuda:5 > output_isruc2.log 2>&1 &

nohup python -u main.py --dataset sleepedf --epoch 100 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --batch_size 32 --device cuda:0 > output_sleepedf1.log 2>&1 &
nohup python -u main.py --dataset sleepedf --epoch 100 --lr 1e-4 --alpha 1e-4 --beta 1e-4 --batch_size 32 --device cuda:1 > output_sleepedf2.log 2>&1 &
nohup python -u main.py --dataset sleepedf --epoch 100 --lr 1e-4 --alpha 1e-5 --beta 1e-5 --batch_size 32 --device cuda:2 > output_sleepedf3.log 2>&1 &

nohup python -u main.py --dataset hmc --epoch 100 --lr 1e-3 --alpha 1e-4 --beta 1e-4 --batch_size 32 --device cuda:3 > output_hmc1.log 2>&1 &
```



For TUH Dataset:
```
nohup python -u main.py --dataset tuab --epoch 50 --lr 1e-3 --device cuda:4 --model_name EEG_Transformer_Network > output_tuab_T.log 2>&1 &

nohup python -u main.py --dataset tuab --epoch 50 --batch_size 32 --lr 1e-3 --device cuda:2 > output_tuab.log 2>&1 &


nohup python main.py --dataset tuev --epoch 50 --batch_size 32 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:3 > output_tuev.log 2>&1 &
nohup python main.py --dataset tuev --epoch 50 --batch_size 32 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --model_name EEG_Transformer_Network --device cuda:3 > output_tuev_T.log 2>&1 &
```




For other:
```
python main.py --dataset seedv --resample True --freq_rate 128 --chunk_second 2 --epoch 50 --lr 1e-5
```

## Performances

### 1. Performances on TUAB



## Citation

TBD