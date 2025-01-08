import sys
import torch
from medmnist import INFO

HOME = '/home/localssk23/'

dataset = sys.argv[1] if len(sys.argv) > 1 else 'breastmnist'
CONFIG = {
   "batch_size": 2048,
   "num_epochs": 100,

   "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
   
   "data_path": f'{HOME}.medmnist/{dataset}.npz',
   "result_path": f'{HOME}localdp/results/',

   "num_folds": 3
}

# # FOR UNIT TESTING ##

# dataset = 'breastmnist'
# CONFIG = {
#    "batch_size": 2,
#    "num_epochs": 1,
#    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#    "data_path": f'/home/localssk23/.medmnist/{dataset}.npz',
#    "result_path": '/home/localssk23/localdp/results/',
#    "num_folds": 1
# }

# #####################

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

data_flag = CONFIG['data_path'].split('/')[-1].split('.')[0]
info = INFO[data_flag]

CONFIG['num_classes'] = len(info['label'])
CONFIG['num_channels'] = info['n_channels']
CONFIG['task'] = info['task']