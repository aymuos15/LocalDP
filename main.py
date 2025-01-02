from train import train
from test import test
from model import Net_28
from load import (
    load_and_prepare_data,
    load_and_prepare_percentage_perturbed_data,
)
from config import CONFIG
from typing import Dict, Tuple
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

def train_and_evaluate(loader_name: str, loaders_dict: Dict[str, Dict]) -> Tuple:
    results = {}
    
    for loader_type, loader_data in loaders_dict.items():
        model = Net_28(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])
        model = train(model, loader_data['loaders'][loader_name], CONFIG['task'])
        acc, auc, class_acc = test(model, loaders_dict['original']['loaders']['test'], CONFIG['task'])
        
        results[loader_type] = {
            'accuracy': acc,
            'auc': auc,
            'class_acc': class_acc
        }
        del model
        
    return results, loaders_dict['original']['class_counts'][loader_name], loaders_dict['original']['class_counts']['test']

def main():
    loaders_dict = {
        'original': {
            'loaders': load_and_prepare_data()[0],
            'class_counts': load_and_prepare_data()[1]
        },
        'noisy_1pixel': {
            'loaders': load_and_prepare_percentage_perturbed_data('one')[0],
            'class_counts': load_and_prepare_percentage_perturbed_data('one')[1]
        },
        'noisy_1%': {
            'loaders': load_and_prepare_percentage_perturbed_data(0.01)[0],
            'class_counts': load_and_prepare_percentage_perturbed_data(0.01)[1]
        },
        'noisy_5%': {
            'loaders': load_and_prepare_percentage_perturbed_data(0.05)[0],
            'class_counts': load_and_prepare_percentage_perturbed_data(0.05)[1]
        },
        'noisy_10%': {
            'loaders': load_and_prepare_percentage_perturbed_data(0.1)[0],
            'class_counts': load_and_prepare_percentage_perturbed_data(0.1)[1]
        },
        'noisy_25%': {
            'loaders': load_and_prepare_percentage_perturbed_data(0.25)[0],
            'class_counts': load_and_prepare_percentage_perturbed_data(0.25)[1]
        },
        'noisy_50%': {
            'loaders': load_and_prepare_percentage_perturbed_data(0.5)[0],
            'class_counts': load_and_prepare_percentage_perturbed_data(0.5)[1]
        },
        'noisy_75%': {
            'loaders': load_and_prepare_percentage_perturbed_data(0.75)[0],
            'class_counts': load_and_prepare_percentage_perturbed_data(0.75)[1]
        },
        'noisy_90%': {
            'loaders': load_and_prepare_percentage_perturbed_data(0.9)[0],
            'class_counts': load_and_prepare_percentage_perturbed_data(0.9)[1]
        },
        'noisy_99%': {
            'loaders': load_and_prepare_percentage_perturbed_data(0.99)[0],
            'class_counts': load_and_prepare_percentage_perturbed_data(0.99)[1]
        }
    }

    dataset_name = CONFIG['data_path'].split('/')[-1].split('.')[0]
    print('DATA: ', dataset_name)
    class_results = []
    overall_results = []
    sample_info = []

    # Get loader types directly from the dictionary
    results, train_class_counts, test_class_counts = train_and_evaluate('Full', loaders_dict)
    
    for label in range(CONFIG['num_classes']):
        result_dict = {
            'Dataset': dataset_name,
            'Split': 'Full',
            'Class': label,
        }
        for loader_type, metrics in results.items():
            result_dict[f'Accuracy_{loader_type}'] = metrics['class_acc'].get(label, 0)
        class_results.append(result_dict)

    overall_dict = {
        'Dataset': dataset_name,
        'Split': 'Full',
    }
    for loader_type, metrics in results.items():
        overall_dict[f'Overall_Accuracy_{loader_type}'] = metrics['accuracy']
        overall_dict[f'Overall_AUC_{loader_type}'] = metrics['auc']
    overall_results.append(overall_dict)

    sample_info.append({
        'Split': 'Full',
        'Total Train Samples': int(sum(train_class_counts)),
        'Total Test Samples': int(sum(test_class_counts)),
        **{f'Train Samples Class {i}': int(train_class_counts[i]) for i in range(CONFIG['num_classes'])},
        **{f'Test Samples Class {i}': int(test_class_counts[i]) for i in range(CONFIG['num_classes'])}
    })

    class_results_df = pd.DataFrame(class_results)
    overall_results_df = pd.DataFrame(overall_results)

    class_results_df.to_csv(CONFIG['result_path'] + dataset_name + '_' + 'class_results.csv', index=False)
    overall_results_df.to_csv(CONFIG['result_path'] + dataset_name + '_' +  'overall_results.csv', index=False)


if __name__ == '__main__':
    main()