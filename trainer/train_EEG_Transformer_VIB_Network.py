import torch
import torch.optim as optim
from .test import simple_test

def train(model, train_loader, test_loader, criterion, optimizer, args):
    model.train()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_class_loss = 0.0
        running_info_loss = 0.0
        correct = 0
        total = 0
        for i, (eeg, label) in enumerate(train_loader):  # 只返回 (eeg, label)
            eeg = eeg.float().to(args.device)
            label = label.to(args.device)

            if args.dataset in ['hinss', 'isruc']:
                eeg = (eeg - eeg.mean(dim=2, keepdim=True)) / eeg.std(dim=2, keepdim=True)
                
            (mu, std), logit = model(eeg)
            class_loss = criterion(logit, label)
            # info_loss = kl_divergence(mu, std)
            # class_loss = F.cross_entropy(logit, label).div(math.log(2))
            info_loss = -0.5*(1+2*std.log()-mu.pow(2)-std.pow(2)).sum(1).mean() # .div(math.log(2))
            total_loss = class_loss + args.beta * info_loss

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

        epoch_loss = running_loss / len(train_loader)
        epoch_class_loss = running_class_loss / len(train_loader)
        epoch_info_loss = running_info_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}, Class Loss: {epoch_class_loss:.4f}, KL Loss: {epoch_info_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        scheduler.step(epoch_loss)
        simple_test(model, test_loader, args)
