import sys
import torch
from medmnist import INFO

dataset = sys.argv[1] if len(sys.argv) > 1 else 'breastmnist'

CONFIG = {
   "batch_size": 1024,
   "num_epochs": 2,
   "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
   "data_path": f'/home/localssk23/.medmnist/{dataset}.npz',
   "result_path": '/home/localssk23/localdp/results/',
   "split_ratio": 0.1
}

data_flag = CONFIG['data_path'].split('/')[-1].split('.')[0]
info = INFO[data_flag]

CONFIG['num_classes'] = len(info['label'])
CONFIG['num_channels'] = info['n_channels']
CONFIG['task'] = info['task']