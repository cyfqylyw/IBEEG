from torcheeg import transforms
from torcheeg.datasets import ISRUCDataset, SleepEDFxDataset

dataset = SleepEDFxDataset(root_path='./datasets/raw/SleepEDF/sleep-edf-database-expanded-1.0.0/', sfreq=100,
                       channels=['EEG Fpz-Cz', 'EEG Pz-Oz'],
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
                       io_path='./datasets/processed/SleepEDF/'
                       )