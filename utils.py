from Datasets import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

def prepare_dataset(args):
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
        dataset = IsrucDataset(data_dir='datasets/raw/ISRUC-mat/Subgroup_1', label_dir='datasets/raw/ISRUC-SLEEP/Subgroup_1', subject_ids=list(range(1, 101)))
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    elif args.dataset == "hinss":
        dataset = HinssDataset()
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    elif args.dataset == "b2014":
        dataset = BNCI2014001_Dataset()
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    elif args.dataset == "b2015":
        dataset = BNCI2015001_Dataset()
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # elif args.dataset == "crowd":
    #     train_dataset = CrowdsourcedDataset(is_train=True)
    #     test_dataset = CrowdsourcedDataset(is_train=False)
    elif args.dataset == "sleepedf":
        dataset = SleepedfDataset()   # 用 torcheeg.datasets.SleepEDFxDataset 读取

        # dataset = SleepEdf_EEGDataset()  # 用 eswa方式 读取
        # print(f"here 1: {len(dataset)}")
        # train_size = int(0.1 * len(dataset))
        # test_size = len(dataset) - train_size
        # dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    elif args.dataset == "hmc":
        dataset = HmcDataset()
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    elif args.dataset == "seedv":
        dataset = SeedVDataset(args)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    elif args.dataset == 'tuab':
        # train_dataset = TuabDataset(args, train=True)
        # test_dataset = TuabDataset(args, train=False)
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)
        return train_loader, test_loader
    elif args.dataset == 'tuev':
        # train_dataset = TuevDataset(args, train=True)
        # test_dataset = TuevDataset(args, train=False)
        train_loader, test_loader, val_loader = prepare_TUEV_dataloader(args)
        return train_loader, test_loader

    print('Dataset information:')
    print(f'\tlen(train_dataset)={len(train_dataset)}')
    print(f'\tlen(test_dataset)={len(test_dataset)}')
    print("*" * 30)

    label_lst = []
    for _, label in test_dataset:
        label_lst.append(label)
    print('unique label in test_dataset:')
    print(np.unique(np.array(label_lst), return_counts=True))
    print("*" * 30)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=0.7, reduction='mean'):  # 原本：alpha=1, gamma=2
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 将 targets 从 [batch_size, 1] 转换为 [batch_size]
        targets = targets.view(-1)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
