import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import softmax

def compute_auc(all_targets, all_outputs, task):
    all_targets, all_outputs = np.array(all_targets), np.array(all_outputs)
    if task == 'multi-label, binary-class':
        return roc_auc_score(all_targets, all_outputs, average='macro')
    elif all_outputs.shape[1] == 2:
        return roc_auc_score(all_targets, all_outputs[:, 1])
    else:
        softmax_outputs = softmax(all_outputs, axis=1)
        return roc_auc_score(all_targets, softmax_outputs, multi_class='ovr', average='macro')