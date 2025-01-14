import numpy as np
import pywt
import torch
import torch.optim as optim
from .test import simple_test
from .nt_xent_loss import nt_xent_loss


def train(model, train_loader, test_loader, criterion, optimizer, args):
    model.train()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_class_loss = 0.0
        running_info_loss = 0.0
        running_cont_loss = 0.0
        correct = 0
        total = 0
        for i, (eeg, label) in enumerate(train_loader):
            if torch.isnan(eeg).any():
                print(f"Warning: NaN detected in batch {i}!")
                continue  # 跳过当前批次

            eeg_ft = torch.abs(torch.fft.fft(eeg, dim=1)).float().to(args.device)
            eeg_wt = torch.Tensor([np.concatenate(pywt.wavedec(subeeg.numpy(), 'db1'), axis=1) for subeeg in eeg]).float().to(args.device)
            if args.dataset in ["isruc", "sleepedf", "hmc", "tuab", "tuev"]:
                eeg_wt = eeg_wt[:,:,:args.chunk_second * args.freq_rate].to(args.device)
            eeg = eeg.float().to(args.device)
            label = label.long().to(args.device)

            (mu, std), logit, eeg_projection, eeg_projection_ft, eeg_projection_wt = model(eeg, eeg_ft, eeg_wt)
            class_loss = criterion(logit, label)
            cont_loss = nt_xent_loss(eeg_projection, eeg_projection_ft, args.temperature) + nt_xent_loss(eeg_projection, eeg_projection_wt, args.temperature)
            info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean() # .div(math.log(2))
            total_loss = class_loss + args.alpha * cont_loss + args.beta * info_loss

            # izy_bound = math.log(10,2) - class_loss
            # izx_bound = info_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

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

