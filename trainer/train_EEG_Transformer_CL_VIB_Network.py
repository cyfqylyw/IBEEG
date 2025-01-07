import numpy as np
import pywt
import torch
import torch.optim as optim
from .test import simple_test
from .nt_xent_loss import nt_xent_loss
from torch.cuda.amp import GradScaler, autocast 


def train(model, train_loader, test_loader, criterion, optimizer, args):
    model.train()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    scaler = GradScaler()  # 初始化 GradScaler，用于缩放损失

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_class_loss = 0.0
        running_info_loss = 0.0
        running_cont_loss = 0.0
        correct = 0
        total = 0
        for i, (eeg, label) in enumerate(train_loader):
            eeg_ft = torch.abs(torch.fft.fft(eeg, dim=1)).float().to(args.device)
            eeg_wt = torch.Tensor([np.concatenate(pywt.wavedec(subeeg.numpy(), 'db1'), axis=1) for subeeg in eeg]).float().to(args.device)
            # eeg_wt = eeg_wt[:eeg_ft.shape[0], :eeg_ft.shape[1], :eeg_ft.shape[2]]
            eeg = eeg.float().to(args.device)
            label = label.long().to(args.device)

            # 使用 autocast 开启混合精度
            # with autocast():
            (mu, std), logit, eeg_projection, eeg_projection_ft, eeg_projection_wt = model(eeg, eeg_ft, eeg_wt)
            class_loss = criterion(logit, label)
            cont_loss = nt_xent_loss(eeg_projection, eeg_projection_ft, args.temperature) + nt_xent_loss(eeg_projection, eeg_projection_wt, args.temperature)
            info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean() # .div(math.log(2))
            total_loss = class_loss + args.alpha * cont_loss + args.beta * info_loss

            # izy_bound = math.log(10,2) - class_loss
            # izx_bound = info_loss

            optimizer.zero_grad()
            scaler.scale(total_loss).backward()  # 缩放损失并反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新缩放器

            # total_loss.backward()
            # optimizer.step()

            _, predicted = torch.max(logit, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            running_loss += total_loss.item()
            running_class_loss += class_loss.item()
            running_info_loss += info_loss.item()
            running_cont_loss += cont_loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_class_loss = running_class_loss / len(train_loader)
        epoch_info_loss = running_info_loss / len(train_loader)
        epoch_cont_loss = running_cont_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}, Class Loss: {epoch_class_loss:.4f}, KL Loss: {epoch_info_loss:.4f}, CL Loss: {epoch_cont_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        scheduler.step(epoch_loss)
        simple_test(model, test_loader, args)

