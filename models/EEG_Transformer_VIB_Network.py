import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numbers import Number
from .base_model import EEG_Encoder, xavier_init

class EEG_Transformer_VIB_Network(nn.Module):
    def __init__(self, args):
        super(EEG_Transformer_VIB_Network, self).__init__()
        self.eeg_encoder = EEG_Encoder(args)

        self.num_channels = args.num_channel if args.selected_channels is None else len(args.selected_channels)
        input_shape = (args.batch_size, self.num_channels, args.chunk_second * args.freq_rate)
        n_feature = self._get_feature_dim(shape=input_shape)
        self.latent_dim = args.latent_dim

        self.fc_statistics = nn.Sequential(
            nn.Linear(n_feature, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.latent_dim))

        # 分类器，输入为采样得到的潜在表示 z
        self.fc_decode = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, args.num_class))

    
    def _get_feature_dim(self, shape):
        x = torch.rand(shape)
        x = self.eeg_encoder(x)
        return x.shape[1]
        
    def reparametrize_n(self, mu, std, device, n=1):
        # reference: https://github.com/1Konny/VIB-pytorch/blob/master/model.py
        def expand(v):
            if isinstance(v, Number):
                return torch.Tensor([v]).expand(n, 1)
            else:
                return v.expand(n, *v.size())

        if n != 1 :
            mu = expand(mu)
            std = expand(std)

        eps = Variable(std.data.new(std.size()).normal_().float().to(device))
        return mu + eps * std
    
    def forward(self, x, num_sample=1):
        x = self.eeg_encoder(x)
        statistics = self.fc_statistics(x)
        mu = statistics[:,:self.latent_dim]
        std = F.softplus(statistics[:,self.latent_dim:]-5, beta=1)
        encoding = self.reparametrize_n(mu, std, device=x.device, n=num_sample)
        logit = self.fc_decode(encoding)

        if num_sample == 1: 
            pass
        elif num_sample > 1: 
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit

    def weight_init(self):
        for m in self._modules:
            xavier_init(m)
