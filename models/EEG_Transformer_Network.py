import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import EEG_Encoder, EEG_Encoder2


class EEG_Transformer_Network(nn.Module):
    def __init__(self, args):
        super(EEG_Transformer_Network, self).__init__()
        self.eeg_encoder = EEG_Encoder(args)

        self.num_channels = args.num_channel if args.selected_channels is None else len(args.selected_channels)
        self.bn = nn.BatchNorm1d(self.num_channels)
        input_shape = (args.batch_size, self.num_channels, args.chunk_second * args.freq_rate)
        n_feature = self._get_feature_dim(shape=input_shape)
        self.fc1 = nn.Linear(n_feature, 128)
        self.fc2 = nn.Linear(128, args.num_class)
    
    def _get_feature_dim(self, shape):
        x = torch.rand(shape)
        x = self.eeg_encoder(x)
        return x.shape[1]
        
    def forward(self, x):
        # x = (x - x.mean(dim=2, keepdim=True)) / x.std(dim=2, keepdim=True)
        x = self.bn(x)
        x = self.eeg_encoder(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
