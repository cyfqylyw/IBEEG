import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import importlib
import time
import random
import scipy.io
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from args import args
from trainer.test import test
from Datasets import *
from torcheeg.datasets import ISRUCDataset
from torcheeg import transforms

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)

t1 = time.time()
print(f"t1 = {t1}")

####### Dataset Preparation #######

if args.dataset == "dreamer":
    dataset = DreamerDataset(args)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
elif args.dataset == "stew":
    dataset = StewDataset(args)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
elif args.dataset == "isruc":
    dataset = IsrucDataset()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
elif args.dataset == "seedv":
    dataset = SeedVDataset(args)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
elif args.dataset == 'tuab':
    train_dataset = TuabDataset(args, train=True)
    test_dataset = TuabDataset(args, train=False)
elif args.dataset == 'tuev':
    train_dataset = TuevDataset(args, train=True)
    test_dataset = TuevDataset(args, train=False)

print('Dataset information:')
print(f'\tlen(train_dataset)={len(train_dataset)}')
print(f'\tlen(test_dataset)={len(test_dataset)}')
print("*" * 30)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

t2 = time.time()
print(f"t2 = {t2}, t2 - t1 = {t2-t1}")

####### Model Training and Test #######

# define model
model_path = 'models.%s' % args.model_name
model = getattr(importlib.import_module(model_path), args.model_name)(args).to(args.device)
if 'VIB' in args.model_name: model.weight_init() 

# define criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# train the model 
train_path = 'trainer.train_%s' % args.model_name
train = getattr(importlib.import_module(train_path), 'train')
train(model, train_loader, test_loader, criterion, optimizer, args)

# evaluate the performance
test(args, model, test_loader, device=args.device)

t3 = time.time()
print(f"t3 = {t3}, t3 - t2 = {t3 - t2}")
