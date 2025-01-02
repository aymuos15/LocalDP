import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from metrics import compute_auc

from config import CONFIG

import tqdm

device = CONFIG['device']

def lr_lambda(epoch):
    initial_lr = 0.001  # Initial learning rate
    if epoch < 50:
        return initial_lr / initial_lr  # Learning rate remains 0.001
    elif epoch < 75:
        return 0.1 * initial_lr / initial_lr  # Delay learning rate to 0.0001 after 50 epochs
    else:
        return 0.01 * initial_lr / initial_lr  # Delay learning rate to 0.00001 after 75 epochs

def train(model, train_loader, task):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []

    if task == "multi-label, binary-class":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss() #! Change to MSE (or just add it)
    criterion.to(device)

    lr = 0.001
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = LambdaLR(optimizer, lr_lambda)

    for epoch in tqdm.tqdm(range(CONFIG['num_epochs'])):
        for inputs, targets in train_loader:
            inputs, targets = inputs.float().to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy for multi-label
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).all(dim=1).sum().item()
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
                
                # Calculate accuracy for standard classification
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

            total += targets.size(0)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Store targets and outputs for AUC calculation
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.detach().cpu().numpy())

    scheduler.step()

    return model


def test(model, test_loader, task):
    model.eval()
    correct = 0
    total = 0
    all_targets = []
    all_outputs = []
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.float().to(device), targets.to(device)

            outputs = model(inputs)

            if task == 'multi-label, binary-class':
                targets = targets.to(float().torch.float32)
                predicted = (outputs > 0.5).float()
                correct += (predicted == targets).all(dim=1).sum().item()
            else:
                targets = targets.squeeze().long()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                
                # Calculate class-wise accuracy
                for true, pred in zip(targets, predicted):
                    if true.ndim == 0:
                        true = int(true.item())
                        pred = int(pred.item())
                    else:
                        true = tuple(true.cpu().numpy())
                        pred = tuple(pred.cpu().numpy())
                    
                    if true == pred:
                        class_correct[true] = class_correct.get(true, 0) + 1
                    class_total[true] = class_total.get(true, 0) + 1

            total += targets.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    acc = 100. * correct / total
    class_acc = {k: 100. * v / class_total[k] for k, v in class_correct.items()}

    auc = compute_auc(all_targets, all_outputs, task)
    
    return acc, auc, class_acc

