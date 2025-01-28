from dataclasses import dataclass
from typing import Dict, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

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

def setup_datasets(
    data: Dict,
    dataset_class: type,
    loader_type: str, 
    transform,
    batch_size: int = 20,
    shuffle: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates train and test datasets using the provided dataset class.
    
    Args:
        data: Dictionary containing train and test data
        dataset_class: The dataset class to instantiate
        loader_type: Type of data loader to use
        transform: Transformations to apply to the data
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    train_dataset = dataset_class(
        data['train_images'],
        data['train_labels'],
        loader_type,
        transform=transform
    )
    
    test_dataset = dataset_class(
        data['test_images'],
        data['test_labels'],
        loader_type,
        transform=transform
    )
    
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    )