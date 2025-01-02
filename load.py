# import torch
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import CONFIG
from dataset import MedMNISTDataset
from dataset import PercentagePerturbedDataset

def split_data(images: np.ndarray, labels: np.ndarray, 
               val_ratio: float = 0.1) -> Tuple[np.ndarray, ...]:
    """Split data into train/val/test sets."""
    train_imgs, test_imgs, train_labs, test_labs = train_test_split(
        images, labels, test_size=CONFIG['split_ratio'], random_state=42
    )
    
    split_idx = int(len(train_imgs) * (1 - val_ratio))
    return (
        train_imgs[:split_idx], train_imgs[split_idx:],
        train_labs[:split_idx], train_labs[split_idx:],
        test_imgs, test_labs
    )

def create_datasets(images: Dict[str, np.ndarray], 
                   labels: Dict[str, np.ndarray],
                   transform) -> Dict[str, MedMNISTDataset]:
    """Create datasets with given images, labels and transform."""
    return {
        name: MedMNISTDataset(imgs, labs, transform=transform)
        for name, (imgs, labs) in zip(
            ['Full', 'LargeSplit', 'SmallSplit', 'test'],
            zip(images.values(), labels.values())
        )
    }

def get_class_counts(labels: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Calculate class distribution for each dataset split."""
    return {
        name: np.bincount(labs.flatten()) 
        for name, labs in labels.items()
    }

def load_base_data():
    """Load and prepare basic dataset without augmentations."""
    data = np.load(CONFIG['data_path'])
    all_images = np.concatenate((data['train_images'], data['test_images']))
    all_labels = np.concatenate((data['train_labels'], data['test_labels']))
    
    train_large, train_small, labels_large, labels_small, test_imgs, test_labs = \
        split_data(all_images, all_labels)
        
    images = {
        'Full': np.concatenate((train_large, train_small)),
        'LargeSplit': train_large,
        'SmallSplit': train_small,
        'test': test_imgs
    }
    
    labels = {
        'Full': np.concatenate((labels_large, labels_small)),
        'LargeSplit': labels_large,
        'SmallSplit': labels_small,
        'test': test_labs
    }
    
    return images, labels

def get_base_transform():
    """Get basic normalization transform."""
    channels = CONFIG['num_channels']
    stats = [0.5] * channels
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=stats, std=stats)
    ])

def prepare_dataloaders(datasets: Dict[str, MedMNISTDataset]) -> Dict[str, DataLoader]:
    """Create dataloaders from datasets."""
    return {
        name: DataLoader(
            dataset,
            batch_size=CONFIG['batch_size'],
            shuffle=name != 'test',
            num_workers=4,
            pin_memory=True
        ) for name, dataset in datasets.items()
    }

def load_and_prepare_data():
    """Main function to load and prepare basic dataset."""
    images, labels = load_base_data()
    transform = get_base_transform()
    datasets = create_datasets(images, labels, transform)
    loaders = prepare_dataloaders(datasets)
    class_counts = get_class_counts(labels)
    
    return loaders, class_counts

# Pixel Perturbation

def load_and_prepare_percentage_perturbed_data(perturbation_percentage: float):
    """Main function to load and prepare percentage-perturbed dataset."""
    images, labels = load_base_data()
    transform = get_base_transform()
    datasets = create_percentage_perturbed_datasets(
        images, labels, transform, perturbation_percentage
    )
    loaders = prepare_dataloaders(datasets)
    class_counts = get_class_counts(labels)
    
    return loaders, class_counts

def create_perturbation_vector(image_shape, num_pixels=1):
    """Create perturbation vector for a single image."""
    height, width = image_shape[:2]
    num_channels = image_shape[2] if len(image_shape) > 2 else 1
    
    # Randomly select pixel positions
    x_pos = np.random.randint(0, height, num_pixels)
    y_pos = np.random.randint(0, width, num_pixels)
    
    # Generate random values for the selected pixels
    if num_channels == 1:
        values = np.random.randint(0, 2, num_pixels)
        perturbation = np.column_stack((x_pos, y_pos, values))
    else:
        values = np.random.randint(0, 2, (num_pixels, num_channels))
        perturbation = np.column_stack((x_pos, y_pos, values))
    
    return perturbation

def create_percentage_perturbed_datasets(
    images: Dict[str, np.ndarray], 
    labels: Dict[str, np.ndarray],
    transform,
    perturbation_percentage: float
) -> Dict[str, PercentagePerturbedDataset]:
    """Create datasets with percentage-wise perturbation."""
    return {
        name: PercentagePerturbedDataset(
            imgs, labs, 
            perturbation_percentage=perturbation_percentage,
            transform=transform
        )
        for name, (imgs, labs) in zip(
            ['Full', 'LargeSplit', 'SmallSplit', 'test'],
            zip(images.values(), labels.values())
        )
    }