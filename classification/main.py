import warnings
from config import CONFIG, PERTURBATION_VALUES, GAUSSIAN_STD_VALUES, REMOVAL_VALUES, DATASET

from typing import Dict

import pandas as pd
import numpy as np

from utils import ModelResults, create_data_transform, setup_datasets
from train import train
from test import test
from model import Net_28

warnings.filterwarnings("ignore")

def train_and_evaluate(loaders_dict: Dict[str, Dict], dataset_style) -> Dict[str, ModelResults]:
    results = {}
    
    for loader_type in loaders_dict.values():
        print(f'Loader Type: {loader_type}')
        
        # Load and prepare data
        data = np.load(CONFIG['data_path'])
        transform = create_data_transform(CONFIG['num_channels'])
        
        # Setup model and datasets
        model = Net_28(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])
        train_loader, test_loader = setup_datasets(data, DATASET[dataset_style], loader_type, transform)
        
        # Train and evaluate
        model_trained = train(model, train_loader, CONFIG['task'])
        acc, auc, class_acc = test(model_trained, test_loader, CONFIG['task'])
        
        results[loader_type] = ModelResults(
            accuracy=acc,
            auc=auc,
            class_acc=class_acc
        )
        
        del model, model_trained
            
    return results

def process_results(results: Dict, dataset_name: str, fold: int, dataset_style: str):
    class_results = []
    overall_results = []
    
    # Process class-wise results
    for label in range(CONFIG['num_classes']):
        result_dict = {'Dataset': dataset_name, 'Class': label}
        for loader_type, metrics in results.items():
            result_dict[f'Accuracy_{loader_type}'] = metrics.class_acc.get(label, 0)
        class_results.append(result_dict)
    
    # Process overall results
    overall_dict = {'Dataset': dataset_name}
    for loader_type, metrics in results.items():
        overall_dict[f'Overall_Accuracy_{loader_type}'] = metrics.accuracy
        overall_dict[f'Overall_AUC_{loader_type}'] = metrics.auc
    overall_results.append(overall_dict)

    # Save results
    pd.DataFrame(class_results).to_csv(
        f"{CONFIG['result_path']}{dataset_name}_class_results_fold_{fold}_{dataset_style}.csv",
        index=False
    )
    pd.DataFrame(overall_results).to_csv(
        f"{CONFIG['result_path']}{dataset_name}_overall_results_fold_{fold}_{dataset_style}.csv",
        index=False
    )

def main(fold: int):
    dataset_name = CONFIG['data_path'].rsplit('/', 1)[-1].rsplit('.', 1)[0]
    print(f'DATA: {dataset_name}')
    print(f'FOLD: {fold}')

    for dataset_style in DATASET.keys():
        print(f'DATASET_STYLE: {dataset_style}')
        if dataset_style == 'Gaussian':
            results = train_and_evaluate(GAUSSIAN_STD_VALUES, dataset_style)
            process_results(results, dataset_name, fold, dataset_style)
        elif dataset_style == 'Perturbed':
            results = train_and_evaluate(PERTURBATION_VALUES, dataset_style)
            process_results(results, dataset_name, fold, dataset_style)
        elif dataset_style == 'Removal':
            results = train_and_evaluate(REMOVAL_VALUES, dataset_style)
            process_results(results, dataset_name, fold, dataset_style)
    print()

if __name__ == '__main__':
    for fold in range(CONFIG['num_folds']):
        main(fold)
        print()
        print()