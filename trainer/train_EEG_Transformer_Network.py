import torch
import torch.optim as optim
from .test import simple_test

def train(model, train_loader, test_loader, criterion, optimizer, args):
    model.train()
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5)
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (eeg, label) in enumerate(train_loader):  # 只返回 (eeg, label)
            eeg = eeg.float().to(args.device)
            label = label.to(args.device)

            if args.dataset in ['hinss', 'isruc']:
                eeg = (eeg - eeg.mean(dim=2, keepdim=True)) / eeg.std(dim=2, keepdim=True)
                
            optimizer.zero_grad()
            outputs = model(eeg)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct / total
        print(f"Epoch [{epoch + 1}/{args.epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        scheduler.step(epoch_loss)
        simple_test(model, test_loader, args)

