import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pywt
import numpy as np
from args import args
from utils import FocalLoss
from Datasets import StewDataset, DreamerDataset
from models.EEG_Transformer_CL_VIB_Network import EEG_Transformer_CL_VIB_Network
from trainer.test import simple_test

device = args.device

if args.dataset == 'stew':
    train_model_path = 'results/ckpt/stew_EEG_Transformer_CL_VIB_Network_50_0.0001_0.001_0.001.pth'
    num_class_test = 5  # 9 for stew, 5 for dreamer
    dataset = DreamerDataset(args)
elif args.dataset == 'dreamer':
    train_model_path = 'results/ckpt/dreamer_EEG_Transformer_CL_VIB_Network_50_0.0001_0.001_0.001.pth'
    num_class_test = 9  # 9 for stew, 5 for dreamer
    dataset = StewDataset(args)

model = EEG_Transformer_CL_VIB_Network(args)
model.load_state_dict(torch.load(train_model_path))
model.to(device)

args.num_class = num_class_test

model.fc_decode = nn.Sequential(
    nn.Linear(model.latent_dim, 512),
    nn.ReLU(True),
    nn.Linear(512, num_class_test)
).to(device)

for param in model.eeg_encoder.parameters():
    param.requires_grad = False
for param in model.eeg_encoder_ft.parameters():
    param.requires_grad = False
for param in model.eeg_encoder_wt.parameters():
    param.requires_grad = False

criterion = FocalLoss(alpha=0.8, gamma=0.7, reduction='mean')
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

def train(model, device, train_loader, optimizer, criterion, num_epochs, args):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (eeg, labels) in enumerate(train_loader):
            eeg_ft = torch.abs(torch.fft.fft(eeg, dim=1)).float().to(args.device)
            eeg_wt = torch.Tensor([np.concatenate(pywt.wavedec(subeeg.numpy(), 'db1'), axis=1) for subeeg in eeg]).float().to(args.device)
            eeg = eeg.float().to(args.device)
            labels = labels.long().to(args.device)

            optimizer.zero_grad()
            _, logits, _, _, _ = model(eeg, eeg_ft, eeg_wt)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
        simple_test(model, test_loader, args)

num_epochs = args.epochs
train(model, device, train_loader, optimizer, criterion, num_epochs, args)