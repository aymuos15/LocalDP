import sys
import torch
from medmnist import INFO
from dataset import MNISTPerturbedDataset, MNISTGaussianDataset, MNISTRemoveDataset

HOME = '/home/localssk23/'

dataset = sys.argv[1] if len(sys.argv) > 1 else 'breastmnist'
# dataset = 'breastmnist'

CONFIG = {
   "batch_size": 2048,
   "num_epochs": 100,

   "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
   
   "data_path": f'{HOME}.medmnist/{dataset}.npz',
   "result_path": f'{HOME}localdp/classification/results/',

   "num_folds": 5
}

DATASET = {
    'Perturbed': MNISTPerturbedDataset,
    'Removal': MNISTRemoveDataset,
    'Gaussian': MNISTGaussianDataset
}

PERTURBATION_VALUES = {
        'original': None,
        'noisy_1pixel': 'one',
        'noisy_1%': 1,
        'noisy_5%': 5,
        'noisy_10%': 10,
        'noisy_25%': 25,
        'noisy_50%': 50,
        'noisy_75%': 75,
        'noisy_90%': 90,
        'noisy_99%': 99
    }

REMOVAL_VALUES = {
        'original': None,
        'remove_1pixel': 'one',
        'remove_1%': 1,
        'remove_5%': 5,
        'remove_10%': 10,
        'remove_25%': 25,
        'remove_50%': 50,
        'remove_75%': 75,
        'remove_90%': 90,
        'remove_99%': 99
    }

GAUSSIAN_STD_VALUES = {
    'gaussian_1': 1,
    'gaussian_5': 5,
    'gaussian_10': 10,
    'gaussian_25': 25,
    'gaussian_50': 50,
    'gaussian_75': 75,
    'gaussian_90': 90,
    'gaussian_99': 99
    }

data_flag = CONFIG['data_path'].split('/')[-1].split('.')[0]
info = INFO[data_flag]

CONFIG['num_classes'] = len(info['label'])
CONFIG['num_channels'] = info['n_channels']
CONFIG['task'] = info['task']
