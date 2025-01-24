# IBEEG

Code implementation for **IBEEG**.



## Requirements

```
conda create -n VIBEEG python=3.9
conda activate VIBEEG
pip install -r requirements.txt
```

## Dataset Preparation and Folder Structure

Download the datasets:
- [DREAMER](https://zenodo.org/records/546113), 
- [Simultaneous Task EEG Workload (STEW)](https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset),
- [ISRUC-SLEEP](https://sleeptight.isr.uc.pt/?page_id=76) 

- [SEED-V Dataset](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html) 

- [Lehner2021](https://www.research-collection.ethz.ch/handle/20.500.11850/458693)
- [Crowdsourced](https://osf.io/9bvgh/), or use the processed version in [EEG2Rep](https://github.com/Navidfoumani/EEG2Rep) with this [link](https://drive.google.com/drive/folders/1KQyST6VJffWWD8r60AjscBy6MHLnT184?usp=sharing)
- [TUAB & TUEV](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/), 


Put them in the folder:
```
IBEEG/
    |_ *.py
    |_ README.md
    |_ requirements.txt
    |_ models/
        |_ [model_name].py
    |_ trainer/
        |_ train_[model_name].py
        |_ test.py
    |_ datasets/
        |_ preprocess_tuab.py
        |_ preprocess_tuev.py
        |_ preprocess_sleepedf.py
        |_ processed/
        |_ raw/
            |_ DREAMER/
                |_ DREAMER.mat
            |_ STEW Dataset/
                |_ *.txt
            |_ ISRUC-SLEEP/
                |_ Subgroup_1/
                    |_ 1/
                        |_ 1.rec
                        |_ 1.txt
            |_ ISRUC-mat/
                |_ Subgroup_1/
                    |_ *.mat
            |_ Hinss2021/
                |_ P01/
                    |_ S1/

            |_ BNCI2014001/
                |_ *.mat

            |_ SleepEDF/
                |_ sleep-edf-database-expanded-1.0.0/
                    |_ sleep-cassette/
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
            |_ Lehner2021/
                |_ Cybathlon_Data/
                    |_ Session [X]/
                        |_ aC/
                            |_ *.eeg
```

Preprocess for TUAB and TUEV datasets.
```
cd datasets/
python preprocess_tuab.py
python preprocess_tuev.py
```

## Usage


```
nohup python -u main.py --dataset dreamer --overlap 0 --epoch 100 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:0 > output_dreamer4.log 2>&1 &
<!-- acc: 0.6845 -->

nohup python -u main.py --dataset stew --overlap 0 --epoch 100 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --device cuda:3 > output_stew.log 2>&1 &
<!-- acc: 0.8000 -->

nohup python -u main.py --dataset isruc --epoch 50 --lr 5e-5 --alpha 1e-3 --beta 1e-4 --batch_size 256 --d_model 256 --device cuda:0 > output_isruc_a-3_b-4_l-3000.log 2>&1 &
<!-- acc: 0.7302 -->

nohup python -u main.py --dataset hinss --epoch 100 --lr 5e-5 --alpha 1e-4 --beta 1e-4 --batch_size 256 --d_model 256 --device cuda:7 > output_hinss.log 2>&1 &
<!-- acc: 0.5164 -->
```


For Sleep Dataset:
```
nohup python -u main.py --dataset b2014 --epoch 100 --lr 5e-5 --alpha 1e-4 --beta 1e-4 --batch_size 256 --d_model 256 --device cuda:0 --model_name EEG_CNN_Network > output_b2014_cnn.log 2>&1 &
nohup python -u main.py --dataset b2014 --epoch 100 --lr 1e-5 --alpha 1e-4 --beta 1e-4 --batch_size 256 --d_model 512 --device cuda:5 --model_name EEG_Transformer_Network > output_b2014_t.log 2>&1 &

nohup python -u main.py --dataset b2015 --epoch 100 --lr 5e-5 --alpha 1e-4 --beta 1e-4 --batch_size 256 --d_model 256 --device cuda:0 --model_name EEG_CNN_Network > output_b2015_cnn.log 2>&1 &
nohup python -u main.py --dataset b2015 --epoch 100 --lr 5e-5 --alpha 1e-4 --beta 1e-4 --batch_size 256 --d_model 256 --device cuda:1 --model_name EEG_Transformer_Network > output_b2015_t.log 2>&1 &

nohup python -u main.py --dataset sleepedf --epoch 50 --lr 1e-4 --alpha 1e-3 --beta 1e-3 --batch_size 32 --device cuda:0 > output_sleepedf1.log 2>&1 &
nohup python -u main.py --dataset sleepedf --epoch 50 --lr 1e-4 --alpha 1e-4 --beta 1e-4 --batch_size 32 --device cuda:5 > output_sleepedf1.log 2>&1 &
nohup python -u main.py --dataset sleepedf --epoch 100 --lr 1e-4 --alpha 1e-5 --beta 1e-5 --batch_size 32 --device cuda:2 > output_sleepedf3.log 2>&1 &

nohup python -u main.py --dataset hmc --epoch 100 --lr 1e-3 --alpha 1e-4 --beta 1e-4 --batch_size 256 --d_model 256 --device cuda:5 > output_hmc_cnn2.log 2>&1 &
```



For TUH Dataset:
```
nohup python -u main.py --dataset tuab --epoch 50 --lr 1e-3 --device cuda:1 --model_name EEG_Transformer_Network > output_tuab_T.log 2>&1 &

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