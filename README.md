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
EEG_embedding
    |_ *.py
    |_ README.md
    |_ datasets
        |_ preprocess_tuab.py
        |_ preprocess_tuev.py
        |_ processed
            |_ TUAB
            |_ TUEV
        |_ raw
            |_ DREAMER
                |_ DREAMER.mat
            |_ STEW Dataset
                |_ *.txt
            |_ SleepEDF
                |_ sleep-edf-database-expanded-1.0.0
                    |_ sleep-telemetry
            |_ ISRUC-SLEEP
                |_ Subgroup_1
                    |_ 1
                        |_ 1.rec
                        |_ 1.txt
                |_ Subgroup_2
                |_ Subgroup_3
            |_ SEED-V
                |_ EEG_raw
            |_ Crowsourced
                |_ Raw Data
            |_ TUAB
                |_ edf
                    |_ train
                    |_ eval
            |_ TUEV
                |_ edf
                    |_ train
                    |_ eval
```

Preprocess for TUAB and TUEV datasets.
```
cd datasets/
python preprocess_tuab.py
python preprocess_tuev.py
```

## Usage

```
nohup python -u main.py --dataset dreamer --overlap 0 --epoch 100 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:0 > output_dreamer2.log 2>&1 &
<!-- acc: 0.6845 -->

nohup python -u main.py --dataset stew --overlap 0 --epoch 100 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:3 > output_stew.log 2>&1 &
<!-- acc: 0.8000 -->

python main.py --dataset isruc --epoch 10 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --batch_size 16

nohup python main.py --dataset tuev --epoch 10 --nhead 5 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:0 > output_tuev.log 2>&1 &

nohup python main.py --dataset tuev --epoch 3 --nhead 10 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:1 --batch_size 16 --model_name EEG_Transformer_Network > output_tuev_EEG_Transformer_Network.log 2>&1 &


python main.py --dataset seedv --resample True --freq_rate 128 --chunk_second 2 --epoch 50 --lr 1e-5

python main.py --dataset tuab --epoch 50 --lr 1e-4 --selected_channels "EEG FP1-REF,EEG F7-REF,EEG T3-REF,EEG T5-REF,EEG O1-REF,EEG FP2-REF,EEG F8-REF,EEG T4-REF,EEG T6-REF,EEG O2-REF,EEG F3-REF,EEG C3-REF,EEG P3-REF,EEG F4-REF,EEG C4-REF,EEG P4-REF"
```

## Performances

### 1. Performances on TUAB



## Citation

TBD