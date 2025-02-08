import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import TemporalConv


class EEG_CNN_Network(nn.Module):
    def __init__(self, args):
        super(EEG_CNN_Network, self).__init__()
        self.temporal_conv = TemporalConv(args) 
        self.conv1 = nn.Conv1d(args.num_channel, 64, kernel_size=3, padding=1)  # 输入通道为 14，输出通道为 32
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dpout1 = nn.Dropout(p=0.2)
        self.dpout2 = nn.Dropout(p=0.2)
        self.dpout3 = nn.Dropout(p=0.2)

        C = args.num_channel if args.selected_channels is None else len(args.selected_channels)
        self.fc1 = nn.Linear(self._get_feature_dim(shape=(args.batch_size, C, args.chunk_second * args.freq_rate)), 128)
        self.fc2 = nn.Linear(128, args.num_class)  # 输出类别数量
    
    def _get_feature_dim(self, shape):
        x = torch.rand(shape)
        x = self.temporal_conv(x) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return x.shape[1]
        
    def forward(self, x):
        x = self.temporal_conv(x) 
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    