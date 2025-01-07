import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from numbers import Number
from .base_model import EEG_Encoder, xavier_init


class EEG_Transformer_CL_VIB_Network(nn.Module):
    def __init__(self, args):
        super(EEG_Transformer_CL_VIB_Network, self).__init__()
        self.eeg_encoder = EEG_Encoder(args)
        self.eeg_encoder_ft = EEG_Encoder(args)
        self.eeg_encoder_wt = EEG_Encoder(args)

        self.num_channels = args.num_channel if args.selected_channels is None else len(args.selected_channels)
        input_shape = (args.batch_size, self.num_channels, args.chunk_second * args.freq_rate)
        n_feature = self._get_feature_dim(shape=input_shape)
        
        self.latent_dim = args.latent_dim

        self.fc_feature_all = nn.Sequential(
            nn.Linear(n_feature * 3, n_feature),
            nn.ReLU(True),
            nn.Linear(n_feature, n_feature))

        self.fc_projector = nn.Sequential(
            nn.Linear(n_feature, n_feature),
            nn.ReLU(True),
            nn.Linear(n_feature, n_feature))
        
        self.fc_projector_ft = nn.Sequential(
            nn.Linear(n_feature, n_feature),
            nn.ReLU(True),
            nn.Linear(n_feature, n_feature))
        
        self.fc_projector_wt = nn.Sequential(
            nn.Linear(n_feature, n_feature),
            nn.ReLU(True),
            nn.Linear(n_feature, n_feature))

        self.fc_statistics = nn.Sequential(
            nn.Linear(n_feature, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2*self.latent_dim))

        # final classifier
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
    
    def forward(self, eeg, eeg_ft, eeg_wt, num_sample=1):
        # feature representation for classification
        eeg_feature = self.eeg_encoder(eeg)
        eeg_feature_ft = self.eeg_encoder_ft(eeg_ft)
        eeg_feature_wt = self.eeg_encoder_wt(eeg_wt)

        # projection for contrastive loss
        eeg_projection = self.fc_projector(eeg_feature)
        eeg_projection_ft = self.fc_projector_ft(eeg_feature_ft)
        eeg_projection_wt = self.fc_projector_wt(eeg_feature_wt)

        # combine all eeg feature 
        # eeg_feature_all = (eeg_feature + eeg_feature_ft + eeg_feature_wt) / 3.0
        eeg_feature_all = self.fc_feature_all(torch.concatenate([eeg_feature, eeg_feature_ft, eeg_feature_wt], dim=1))

        statistics = self.fc_statistics(eeg_feature_all)
        mu = statistics[:,:self.latent_dim]
        std = F.softplus(statistics[:,self.latent_dim:]-5, beta=1)
        encoding = self.reparametrize_n(mu, std, device=eeg_feature_all.device, n=num_sample)
        logit = self.fc_decode(encoding)

        if num_sample == 1: 
            pass
        elif num_sample > 1: 
            logit = F.softmax(logit, dim=2).mean(0)

        return (mu, std), logit, eeg_projection, eeg_projection_ft, eeg_projection_wt

    def weight_init(self):
        for m in self._modules:
            xavier_init(m)
