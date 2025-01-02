import torch

from medmnist import INFO

CONFIG = {
    "batch_size": 1024,
    "num_epochs": 2,
    "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "data_path": '/home/localssk23/.medmnist/breastmnist.npz',
    "result_path": '/home/localssk23/localdp/results/',
    "split_ratio": 0.1
}

# get data_flag from data_path
data_flag = CONFIG['data_path'].split('/')[-1].split('.')[0]
info = INFO[data_flag]

# get the number of classes from the INFO dictionary
CONFIG['num_classes'] = len(info['label'])
CONFIG['num_channels'] = info['n_channels']
CONFIG['task'] = info['task']
