import torch

from metrics import compute_auc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
