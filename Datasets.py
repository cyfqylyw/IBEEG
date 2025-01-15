import os
import mne
import numpy as np
import pandas as pd
import scipy.io
import pickle
import glob
from collections import defaultdict
from scipy.signal import resample

import torchvision

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

import torcheeg
from torcheeg import transforms
from torcheeg.datasets import SleepEDFxDataset, HMCDataset

from transform import *
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import warnings
warnings.filterwarnings("ignore")


class BaseEEGDataset(Dataset):
    """
    base class for EEG dataset
    """
    def __init__(self):
        super().__init__()
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eeg_data, label = self.samples[idx]
        return torch.tensor(eeg_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class DreamerDataset(BaseEEGDataset):
    def __init__(self, args, data_path='datasets/raw/DREAMER/DREAMER.mat'):
        super().__init__()
        self.data = scipy.io.loadmat(data_path)
        self.chunk_second = args.chunk_second
        self.freq_rate = args.freq_rate
        self.chunk_length = args.chunk_second * args.freq_rate  # 2s * 128 = 256
        self.overlap = args.overlap
        self.selected_channels = args.selected_channels
        self.electrode_lst = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        """
        Prepare the samples by splitting the data into chunks of size (chunk_length, len(selected_channels)).T.
        """
        samples = []

        # Loop over all participants and films
        for participant in range(23):
            for film in range(18):
                # Extract EEG data for a specific participant and film
                eeg_data = self.data['DREAMER'][0, 0]['Data'][0, participant]['EEG'][0, 0]['stimuli'][0, 0][film, 0]
                arousal = self.data['DREAMER'][0, 0]['Data'][0, participant]['ScoreArousal'][0, 0][film, 0]
                
                # Select the channels if provided
                if self.selected_channels:
                    selected_indices = [self.electrode_lst.index(channel) for channel in self.selected_channels]
                    eeg_data = eeg_data[:, selected_indices]  # Select the required channels
                
                # Get the total time length of the EEG data (first dimension)
                time_length = eeg_data.shape[0]
                step_size = self.chunk_length - self.overlap

                for i in range((time_length - self.chunk_length) // step_size + 1):
                    start_idx = i * step_size
                    end_idx = start_idx + self.chunk_length
                    chunk = eeg_data[start_idx:end_idx, :]  # Get the chunk
                    samples.append((chunk.T, arousal - 1))
        
        return samples


class StewDataset(BaseEEGDataset):
    def __init__(self, args, data_path="datasets/raw/STEW Dataset/"):
        super().__init__()
        self.data_path = data_path
        self.chunk_second = args.chunk_second
        self.freq_rate = args.freq_rate
        self.chunk_length = args.chunk_second * args.freq_rate  # 2s * 128 = 256
        self.overlap = args.overlap
        self.selected_channels = args.selected_channels
        self.electrode_lst = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'] # the same as Dreamer dataset
        self.samples = self._prepare_samples()
    
    def _prepare_samples(self):
        """
        Prepare the samples by reading .txt files and splitting the data into chunks of size (chunk_length, len(selected_channels)).T.
        """
        samples = []
        label_df = pd.read_csv(os.path.join(self.data_path, 'ratings.txt'), names=['sub_id', 'lo', 'hi'], index_col=None)
        # Loop through all 96 txt files in the data folder
        for filename in os.listdir(self.data_path):
            if filename.endswith('.txt') and filename != "ratings.txt":
                sub_id, lh_flag = int(filename[3:5]), filename[6:8]
                if sub_id in [5, 24, 42]:
                    continue
                label = label_df[label_df['sub_id'] == sub_id][lh_flag].item()

                file_path = os.path.join(self.data_path, filename)
                eeg_data = np.loadtxt(file_path)

                # Select the channels if specified
                if self.selected_channels:
                    selected_indices = [self.electrode_lst.index(channel) for channel in self.selected_channels]
                    eeg_data = eeg_data[:, selected_indices]  # Select the required channels

                # Get the total time length of the EEG data (first dimension)
                time_length = eeg_data.shape[0]
                step_size = self.chunk_length - self.overlap

                # Loop to split data into chunks
                for i in range((time_length - self.chunk_length) // step_size + 1):
                    start_idx = i * step_size
                    end_idx = start_idx + self.chunk_length
                    chunk = eeg_data[start_idx:end_idx, :]  # Get the chunk
                    
                    # Assuming that arousal (or other labels) are stored in the filename as 'hi' or 'lo'
                    # label = 1 if 'hi' in filename else 0  # High arousal (hi) or Low arousal (lo)  # 如果是这样，num_calss = 2
                    
                    samples.append((chunk.T, label-1))  # Add sample (transposed chunk, label)
        
        return samples



class IsrucDataset(Dataset):
    def __init__(self, data_dir='datasets/raw/ISRUC-mat/Subgroup_1', label_dir='datasets/raw/ISRUC-SLEEP/Subgroup_1', subject_ids=list(range(1, 101))):
        """
        Args:
            data_dir (str): 包含 .mat 文件的目录路径。
            label_dir (str): 包含标签文件的目录路径。
            subject_ids (list): 要加载的 subject id 列表。
        """
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.subject_ids = subject_ids
        self.data = []
        self.labels = []

        # 加载所有数据
        for subject_id in subject_ids:
            # 加载 .mat 文件
            mat_file = os.path.join(data_dir, f'subject{subject_id}.mat')
            mat_data = scipy.io.loadmat(mat_file)
            
            # 提取6个通道的数据
            channels = ['F3_A2', 'C3_A2', 'O1_A2', 'F4_A1', 'C4_A1', 'O2_A1']
            eeg_data = np.stack([mat_data[channel] for channel in channels], axis=1)  # (num, 6, 6000)
            eeg_data = eeg_data[:, :, ::5]    # (num, 6, 3000) (num, 6, 1200)
            
            # 加载标签文件
            label_file = os.path.join(label_dir, f'{subject_id}/{subject_id}_1.txt')
            with open(label_file, 'r') as f:
                labels = f.read().splitlines()
                labels = np.array([int(label) for label in labels[:eeg_data.shape[0]]])  # 抛弃最后30个标签
            
            # 将标签中的5改为4
            labels[labels == 5] = 4
            
            # 将数据和标签添加到列表中
            self.data.append(eeg_data)
            self.labels.append(labels)
            
        self.data = np.concatenate(self.data, axis=0)  # (total_num, 6, 6000)
        self.labels = np.concatenate(self.labels, axis=0)  # (total_num,)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return data, label


class HmcDataset(Dataset):
    def __init__(self):
        self.original_dataset = HMCDataset(root_path='./datasets/raw/HMC/physionet.org/files/hmc-sleep-staging/1.1/recordings',
                    sfreq=100,
                    channels=['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2'],
                    label_transform=transforms.Compose([
                        transforms.Select('label'),
                        transforms.Mapping({'Sleep stage W': 0,
                                            'Sleep stage N1': 1,
                                            'Sleep stage N2': 2,
                                            'Sleep stage N3': 3,
                                            'Sleep stage R': 4,
                                            'Lights off@@EEG F4-A1': 0})
                    ]),
                    online_transform=transforms.ToTensor(),
                    io_path='./datasets/processed/HMC/'
                    )
        
        # train_size = int(0.1 * len(self.original_dataset))
        # test_size = len(self.original_dataset) - train_size
        # self.original_dataset, _ = random_split(self.original_dataset, [train_size, test_size])
        self.data, self.labels = self._extract_data_and_labels()
    
    def _extract_data_and_labels(self):
        data = []
        labels = []
        for sample in self.original_dataset:
            data.append(sample[0].numpy())  # Extract EEG data
            labels.append(sample[1])        # Extract label
        return np.array(data), np.array(labels)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the resampled data and labels
        eeg_data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return eeg_data, label


class SleepEdf_EEGDataset(BaseEEGDataset):
    def __init__(self, data_path="./datasets/processed/sleepedf", channel="Fpz-Cz"):
        self.data_path = data_path
        self.channel = channel
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        dirname = os.path.join(self.data_path, self.channel)
        dirlst = [x for x in os.listdir(dirname) if x[-3:] == 'npz']
        for fn in dirlst:
            filepath = os.path.join(dirname, fn)
            data = np.load(filepath)
            for i in range(len(data['y'])):
                sample_data = torch.from_numpy(data['x'][i]).float().unsqueeze(0)
                sample_label = torch.tensor(data['y'][i], dtype=torch.long).unsqueeze(0)
                samples.append((sample_data, sample_label))
        
        print(f'here 3: len(samples)')

        return samples


class SleepedfDataset(Dataset):
    def __init__(self):
        self.original_dataset = SleepEDFxDataset(root_path='./datasets/raw/SleepEDF/sleep-edf-database-expanded-1.0.0/', 
            sfreq=100,
            channels=['EEG Fpz-Cz', 'EEG Pz-Oz'],
            label_transform=transforms.Compose([
                transforms.Select('label'),
                transforms.Mapping({'Sleep stage W': 0, 'Sleep stage N1': 1, 'Sleep stage N2': 2, 'Sleep stage N3': 3, 'Sleep stage R': 4, 'Lights off@@EEG F4-A1': 0})
                ]),
                online_transform=transforms.ToTensor(),
                io_path='./datasets/processed/SleepEDF/'
            )
        
        train_size = int(0.1 * len(self.original_dataset))
        test_size = len(self.original_dataset) - train_size
        self.original_dataset, _ = random_split(self.original_dataset, [train_size, test_size])
        self.data, self.labels = self._extract_data_and_labels()
        self.data, self.labels = self._apply_undersampling()
    
    def _extract_data_and_labels(self):
        data = []
        labels = []
        for sample in self.original_dataset:
            data.append(sample[0].numpy())  # Extract EEG data
            labels.append(sample[1])        # Extract label
        return np.array(data), np.array(labels)
    
    def _apply_smote(self):
        # Reshape data to 2D for SMOTE (samples, features)
        n_samples, n_channels, n_timesteps = self.data.shape
        data_reshaped = self.data.reshape(n_samples, -1)  # Flatten the EEG data
        smote = SMOTE(random_state=42)
        data_resampled, labels_resampled = smote.fit_resample(data_reshaped, self.labels)
        data_resampled = data_resampled.reshape(-1, n_channels, n_timesteps)
        return data_resampled, labels_resampled

    def _apply_undersampling(self):
        n_samples, n_channels, n_timesteps = self.data.shape
        data_reshaped = self.data.reshape(n_samples, -1)  # Flatten the EEG data
        rus = RandomUnderSampler(random_state=42)
        data_resampled, labels_resampled = rus.fit_resample(data_reshaped, self.labels)
        data_resampled = data_resampled.reshape(-1, n_channels, n_timesteps)
        return data_resampled, labels_resampled

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return the resampled data and labels
        eeg_data = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return eeg_data, label


# class CrowdsourcedDataset(Dataset):
#     def __init__(self, datapath='datasets/raw/Crowdsourced-npy/Crowdsource.npy', is_train=True):
#         super().__init__()
#         self.datapath = datapath
#         self.is_train = is_train

#         data = np.load(datapath, allow_pickle=True)

#         if self.is_train:
#             self.data = data.item().get('All_train_data')    # (10403, 14, 256)
#             self.label = data.item().get('All_train_label')  # (10403,)
#         else:
#             self.data = data.item().get('test_data')         # (1893, 14, 256)
#             self.label = data.item().get('test_label')       # (1893,)
    
#     def __len__(self):
#         return len(self.label)
    
#     def __getitem__(self, index):
#         sample = self.data[index]
#         label = self.label[index]
#         return sample, label
    

# class IsrucDataset_torcheeg_load(Dataset):
#     def __init__(self):
#         self.original_dataset = torcheeg.datasets.ISRUCDataset(root_path='./datasets/raw/ISRUC-SLEEP', sfreq=100,
#             channels=['F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1'],
#             label_transform=transforms.Compose([
#                 transforms.Select('label'),
#                 transforms.Mapping({'Sleep stage W': 0,
#                                     'Sleep stage N1': 1,
#                                     'Sleep stage N2': 2,
#                                     'Sleep stage N3': 3,
#                                     'Sleep stage R': 4,
#                                     'Lights off@@EEG F4-A1': 5})
#             ]),
#             # online_transform=transforms.Compose([
#             #             transforms.MeanStdNormalize(),
#             #             transforms.ToTensor(),
#             #         ]),
#             # io_path='./datasets/processed/ISRUC-SLEEP_w_MeanStdNormalize/'

#             online_transform=transforms.ToTensor(),
#             io_path='./datasets/processed/ISRUC-SLEEP5'

#             # online_transform=torchvision.transforms.Compose([        # Mapping: Lights off: 5
#             #     torchvision.transforms.ToTensor(),
#             #     torchvision.transforms.Normalize((0.1307,), (0.3081,))
#             # ]),
#             # io_path='./datasets/processed/ISRUC-SLEEP_tvtrans'
#             )
        
#         train_size = int(0.1 * len(self.original_dataset))
#         test_size = len(self.original_dataset) - train_size
#         self.original_dataset, _ = random_split(self.original_dataset, [train_size, test_size])

#         self.valid_indices = self._get_valid_indices()
#         self.data, self.labels = self._extract_data_and_labels()
#         # self.data, self.labels = self._apply_undersampling()

#     def _get_valid_indices(self):
#         valid_indices = []
#         for i in range(len(self.original_dataset)):
#             a = self.original_dataset[i]
#             if (a[0] != 0).all():
#                 valid_indices.append(i)
#         return valid_indices
    
#     def _extract_data_and_labels(self):
#         data = []
#         labels = []
#         for idx in self.valid_indices:
#             sample = self.original_dataset[idx]
#             data.append(sample[0].numpy())  # Extract EEG data
#             labels.append(sample[1])        # Extract label
#         return np.array(data), np.array(labels)
    
#     def _apply_undersampling(self):
#         # Reshape data to 2D for RandomUnderSampler (samples, features)
#         n_samples, n_channels, n_timesteps = self.data.shape
#         data_reshaped = self.data.reshape(n_samples, -1)  # Flatten the EEG data
        
#         # Apply RandomUnderSampler
#         rus = RandomUnderSampler(random_state=42)
#         data_resampled, labels_resampled = rus.fit_resample(data_reshaped, self.labels)
        
#         # Reshape data back to original shape
#         data_resampled = data_resampled.reshape(-1, n_channels, n_timesteps)
#         return data_resampled, labels_resampled

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         # original_idx = self.valid_indices[idx]
#         # return self.original_dataset[original_idx]
#         eeg_data = torch.tensor(self.data[idx], dtype=torch.float32)
#         label = torch.tensor(self.labels[idx], dtype=torch.long)
#         return eeg_data, label
    

class SeedVDataset(BaseEEGDataset):
    def __init__(self, args, data_path="datasets/raw/SEED-V/EEG_raw/"):
        super().__init__()
        self.data_path = data_path
        self.chunk_second = args.chunk_second
        self.freq_rate = args.freq_rate
        self.chunk_length = self.chunk_second * self.freq_rate  # e.g., 1s * 1000 = 1000
        self.overlap = args.overlap
        self.resample = args.resample
        self.selected_channels = args.selected_channels
        self.electrode_lst = [
            'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2',
            'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4', 'Fz', 'Cz',
            'Pz', 'Oz', 'CPz', 'Fp1', 'Fp2', 'FC1', 'FC2', 'C1',
            'C2', 'CP1', 'CP2', 'P1', 'P2', 'POz', 'O9', 'O10',
            'F1', 'F2', 'F5', 'F6', 'F9', 'F10', 'C5', 'C6',
            'T9', 'T10', 'P5', 'P6', 'PO3', 'PO4', 'PO7', 'PO8'
        ]  # Adjust based on SEED-V electrode configuration
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        """
        Prepare the samples by reading .cnt files, slicing based on session-specific time windows,
        and splitting the data into chunks with the specified overlap.
        """
        samples = []

        session_times = {
            '1': {
                'start': [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
                'end':   [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359],
                'labels': [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0]
            },
            '2': {
                'start': [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
                'end':   [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817],
                'labels': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]
            },
            '3': {
                'start': [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
                'end':   [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066],
                'labels': [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]
            }
        }

        # Iterate through all .cnt files in the data directory
        for filename in os.listdir(self.data_path)[:1]:
            print(filename)
            if filename.endswith('.cnt'):
                file_path = os.path.join(self.data_path, filename)
                
                parts = filename.split('.')[0].split('_')
                _, session, _ = parts

                try:
                    eeg_raw = mne.io.read_raw_cnt(file_path, preload=True, verbose=False)
                except Exception as e:
                    print(f"Error reading {file_path}: {e}. Skipping.")
                    continue

                # Drop the useless channels
                useless_ch = ['M1', 'M2', 'VEO', 'HEO']
                existing_useless_ch = [ch for ch in useless_ch if ch in eeg_raw.ch_names]
                eeg_raw.drop_channels(existing_useless_ch)

                # Select the desired channels if specified
                if self.selected_channels:
                    # Ensure all selected channels are present
                    selected = [ch for ch in self.selected_channels if ch in eeg_raw.ch_names]
                    if not selected:
                        print(f"No selected channels found in {filename}. Skipping.")
                        continue
                    eeg_raw.pick_channels(selected)

                if self.resample:
                    # resample to specific frequency rate (self.freq_rate)
                    eeg_raw.resample(self.freq_rate, npad="auto")

                data_matrix = eeg_raw.get_data()  # Shape: (num_channel, time_length)
                start_seconds = session_times[session]['start']
                end_seconds = session_times[session]['end']
                labels = session_times[session]['labels']

                # Iterate through each of the 15 segments
                for i in range(15):
                    start_sec = start_seconds[i]
                    end_sec = end_seconds[i]
                    label = labels[i]

                    start_idx = int(start_sec * self.freq_rate)
                    end_idx = int(end_sec * self.freq_rate)

                    segment_data = data_matrix[:, start_idx:end_idx]  # Shape: (n_channels, segment_length)
                    segment_length = segment_data.shape[1]
                    step_size = self.chunk_length - self.overlap
                    num_chunks = (segment_length - self.chunk_length) // step_size + 1

                    # Split the segment into chunks
                    for j in range(num_chunks):
                        start_chunk = j * step_size
                        end_chunk = start_chunk + self.chunk_length
                        chunk = segment_data[:, start_chunk:end_chunk]

                        samples.append((chunk, label))

        return samples


# class TuevDataset(BaseEEGDataset):
#     def __init__(self, args, data_path='datasets/processed/TUEV/edf/', train=True):
#         super().__init__()
#         self.data_path = data_path
#         self.mode = 'train' if train else 'eval'
#         self.folder_path = os.path.join(self.data_path, self.mode)
#         self.samples = self._prepare_samples()
    
#     def _prepare_samples(self):
#         samples = []

#         for filename in os.listdir(self.folder_path):
#             if filename[-4:] == ".pkl":
#                 data = pickle.load(open(os.path.join(self.folder_path, filename), 'rb'))
#                 chunk, label = data['signal'], data['label'][0]
#                 samples.append((chunk, label-1))

#         return samples
    

class TuevDataset(Dataset):
    def __init__(self, args, data_path='datasets/processed/TUEV/edf/', train=True):
        super().__init__()
        self.data_path = data_path
        self.mode = 'train' if train else 'eval'
        self.folder_path = os.path.join(self.data_path, self.mode)
        self.filenames = [f for f in os.listdir(self.folder_path) if f.endswith(".pkl")]
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        data = pickle.load(open(os.path.join(self.folder_path, filename), 'rb'))
        chunk, label = data['signal'], data['label'][0]
        return chunk, int(label - 1)


class TUABLoader(torch.utils.data.Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["X"]
        # from default 200Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 10 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = sample["y"]
        X = torch.FloatTensor(X)
        return X, Y


def prepare_TUAB_dataloader(args):
    root = "datasets/processed/TUAB"

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    train_files = train_files[:100000]
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    print('length of train/val/test files:')
    print(len(train_files), len(val_files), len(test_files))

    # prepare training and test data loader
    train_loader = DataLoader(
        TUABLoader(os.path.join(root, "train"), train_files),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    print('length of train/val/test data loader:')
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


class TUEVLoader(Dataset):
    def __init__(self, root, files, sampling_rate=200):
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        X = sample["signal"]
        # 256 * 5 -> 1000, from 256Hz to ?
        if self.sampling_rate != self.default_rate:
            X = resample(X, 5 * self.sampling_rate, axis=-1)
        X = X / (
            np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
            + 1e-8
        )
        Y = int(sample["label"][0] - 1)
        X = torch.FloatTensor(X)
        return X, Y


def prepare_TUEV_dataloader(args):
    root = "./datasets/processed/TUEV/edf"
    train_files = os.listdir(os.path.join(root, "train"))
    train_sub = list(set([f.split("_")[0] for f in train_files]))
    print("train sub", len(train_sub))
    test_files = os.listdir(os.path.join(root, "eval"))

    val_sub = np.random.choice(train_sub, size=int(len(train_sub) * 0.1), replace=False)
    train_sub = list(set(train_sub) - set(val_sub))
    val_files = [f for f in train_files if f.split("_")[0] in val_sub]
    train_files = [f for f in train_files if f.split("_")[0] in train_sub]

    # prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUEVLoader(os.path.join(root, "train"), train_files),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUEVLoader(os.path.join(root, "eval"), test_files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUEVLoader(os.path.join(root, "train"), val_files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )

    print('length of train/val/test files:')
    print(len(train_files), len(val_files), len(test_files))

    print('length of train/val/test data loader:')
    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, test_loader, val_loader


# class TuabDataset(BaseEEGDataset):
#     def __init__(self, args, data_path="datasets/raw/TUAB/edf/", train=True):
#         super().__init__()
#         self.data_path = data_path
#         self.mode = 'train' if train else 'eval'
#         self.chunk_second = args.chunk_second
#         self.freq_rate = args.freq_rate
#         self.chunk_length = args.chunk_second * args.freq_rate  # 2s * 128 = 256
#         self.overlap = args.overlap
#         self.selected_channels = args.selected_channels
#         self.samples = self._prepare_samples()
    
#     def _prepare_samples(self):
#         """
#         Prepare the samples by reading .edf files and splitting the data into chunks of size (chunk_length, len(selected_channels)).
#         """
#         samples = []
#         folder_path = os.path.join(self.data_path, self.mode)  # Folder for training or evaluation

#         # Loop through abnormal and normal folders within train or eval directories
#         for label_folder in os.listdir(folder_path):
#             label_path = os.path.join(folder_path, label_folder)
#             tcp_path = os.path.join(label_path, '01_tcp_ar')
            
#             # Check if it's a folder and contains EDF files
#             if os.path.isdir(label_path):
#                 label = 0 if 'abnormal' in label_folder else 1

#                 # Loop through all EDF files in the current folder
#                 for filename in os.listdir(tcp_path):
#                     if filename.endswith('.edf'):
#                         file_path = os.path.join(tcp_path, filename)
#                         raw_data = mne.io.read_raw_edf(file_path)
                        
#                         # Select the channels if specified
#                         if self.selected_channels:
#                             selected_indices = [raw_data.info['ch_names'].index(channel) for channel in self.selected_channels]
#                             eeg_data = raw_data.get_data(picks=selected_indices)  # Get selected channels
#                         else:
#                             eeg_data = raw_data.get_data()  # Use all channels

#                         time_length = eeg_data.shape[1]
#                         step_size = self.chunk_length - self.overlap

#                         for i in range((time_length - self.chunk_length) // step_size + 1):
#                             start_idx = i * step_size
#                             end_idx = start_idx + self.chunk_length
#                             chunk = eeg_data[:, start_idx:end_idx]  # Get the chunk
#                             # print(chunk.shape)
                            
#                             # Append sample (transposed chunk, label)
#                             samples.append((chunk, label))

#         return samples
