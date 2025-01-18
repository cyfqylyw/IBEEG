import torch
import numpy as np
import importlib
import time
import torch.nn as nn
import torch.optim as optim

from args import args
from trainer.test import test
from utils import prepare_dataset, FocalLoss

import warnings
warnings.filterwarnings('ignore')

seed = 3407
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

t1 = time.time()
print(f"t1 = {t1}")

####### Dataloader Preparation #######

train_loader, test_loader = prepare_dataset(args)

t2 = time.time()
print(f"t2 = {t2}, t2 - t1 = {t2-t1}")

####### Model Training and Test #######

# define model
model_path = 'models.%s' % args.model_name
model = getattr(importlib.import_module(model_path), args.model_name)(args).to(args.device)
if 'VIB' in args.model_name: model.weight_init() 

# define criterion and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = FocalLoss(alpha=0.8, gamma=0.7, reduction='mean')
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# train the model 
train_path = 'trainer.train_%s' % args.model_name
train = getattr(importlib.import_module(train_path), 'train')
train(model, train_loader, test_loader, criterion, optimizer, args)

# evaluate the performance
test(args, model, test_loader, device=args.device)

t3 = time.time()
print(f"t3 = {t3}, t3 - t2 = {t3 - t2}")
