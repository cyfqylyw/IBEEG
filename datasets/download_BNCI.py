import os
import mne
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from moabb.datasets import BNCI2014_001, BNCI2015_001
from moabb.paradigms import MotorImagery

dsname = 'BNCI2014001'     # BNCI2014001,  BNCI2015001
dataset = BNCI2014_001()   # BNCI2014_001, BNCI2015_001


n_cls_dict = {'BNCI2014001': 4, 'BNCI2015001': 2}
n_class = n_cls_dict[dsname]

paradigm = MotorImagery(n_classes=n_class)

X, y, metadata = paradigm.get_data(dataset)

if dsname == 'BNCI2014001':
    label_mapping = {
        "left_hand": 0,
        "right_hand": 1,
        "feet": 2,
        "tongue": 3
    }
else:
    label_mapping = {
        "right_hand": 0,
        "feet": 1
    }


y_int = np.array([label_mapping[label] for label in y])

os.makedirs(f'./processed/{dsname}/', exist_ok=True)
np.save(f"./processed/{dsname}/X_data.npy", X)
np.save(f"./processed/{dsname}/y_labels.npy", y_int)