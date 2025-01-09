from einops import rearrange
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from numbers import Number


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


class TemporalConv(nn.Module):
    """ 
    Image to Patch Embedding

    Input: x.shape = torch.rand((batch_size, num_channel, chunk_second * freq_rate))
    Output: shape = torch.rand((batch_size, num_channel, chunk_second * freq_rate))
    """
    def __init__(self, args, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()
        self.linear = nn.Linear(self._forward_for_one_time(args), args.chunk_second * args.freq_rate)
    
    def _forward_for_one_time(self, args):
        x = torch.rand((args.batch_size, args.num_channel, args.chunk_second * args.freq_rate))
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x.shape[-1]

    def forward(self, x, **kwargs):
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        x = self.linear(x)
        return x

class EEG_Encoder(nn.Module):
    def __init__(self, args):
        super(EEG_Encoder, self).__init__()
        self.temporal_conv = TemporalConv(args)
        self.num_channels = args.num_channel if args.selected_channels is None else len(args.selected_channels)
        self.batch_norm_conv = nn.BatchNorm1d(self.num_channels)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=args.chunk_second * args.freq_rate, nhead=args.nhead, dim_feedforward=2048, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        self.dropout_transformer = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.batch_norm_conv(x)
        x = self.transformer_encoder(x)
        x = self.dropout_transformer(x)
        x = x.view(x.size(0), -1)
        return x
    
