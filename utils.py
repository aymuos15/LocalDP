from dataclasses import dataclass
from typing import Dict

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import MNISTPerturbedDataset
from train import train
from test import test
from model import Net_28

from config import CONFIG

@dataclass
class ModelResults:
    accuracy: float
    auc: float
    class_acc: Dict[int, float]

def create_data_transform(num_channels: int) -> transforms.Compose:
    stats = [0.5] * num_channels
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=stats, std=stats)
    ])

def setup_datasets(data: Dict, loader_type: str, transform) -> tuple[DataLoader, DataLoader]:
    train_dataset = MNISTPerturbedDataset(
        data['train_images'], 
        data['train_labels'], 
        loader_type, 
        transform=transform
    )
    
    test_dataset = MNISTPerturbedDataset(
        data['test_images'], 
        data['test_labels'], 
        loader_type, 
        transform=transform
    )
    
    return (
        DataLoader(train_dataset, batch_size=20, shuffle=False),
        DataLoader(test_dataset, batch_size=20, shuffle=False)
    )

def train_and_evaluate(loaders_dict: Dict[str, Dict]) -> Dict[str, ModelResults]:
    results = {}
    
    for loader_type in loaders_dict.values():
        print(f'Loader Type: {loader_type}')
        
        # Load and prepare data
        data = np.load(CONFIG['data_path'])
        transform = create_data_transform(CONFIG['num_channels'])
        
        # Setup model and datasets
        model = Net_28(CONFIG['num_channels'], CONFIG['num_classes']).to(CONFIG['device'])
        train_loader, test_loader = setup_datasets(data, loader_type, transform)
        
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

def process_results(results: Dict, dataset_name: str, fold: int):
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
        f"{CONFIG['result_path']}{dataset_name}_class_results_fold_{fold}.csv",
        index=False
    )
    pd.DataFrame(overall_results).to_csv(
        f"{CONFIG['result_path']}{dataset_name}_overall_results_fold_{fold}.csv",
        index=False
    )
