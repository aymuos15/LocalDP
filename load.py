import torch
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import CONFIG
from dataset import (
    MedMNISTDataset,
    GaussianNoiseTransform,
    PoissonNoiseTransform,
    SaltPepperNoiseTransform, 
    SpeckleNoiseTransform,
    perturb_image_torch
)

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

def load_and_prepare_noisy_data(noise_type: str = 'gaussian', **noise_params):
    """Load and prepare dataset with noise augmentation."""
    noise_transforms = {
        'gaussian': GaussianNoiseTransform,
        'salt_pepper': SaltPepperNoiseTransform,
        'poisson': PoissonNoiseTransform,
        'speckle': SpeckleNoiseTransform
    }
    
    if noise_type not in noise_transforms:
        raise ValueError(f"Unsupported noise type: {noise_type}")
    
    images, labels = load_base_data()
    noise_transform = noise_transforms[noise_type](**noise_params)
    datasets = create_datasets(images, labels, noise_transform)
    loaders = prepare_dataloaders(datasets)
    class_counts = get_class_counts(labels)
    
    return loaders, class_counts

def load_and_prepare_data_one_pixel():
    data = np.load(CONFIG['data_path'])
    all_images = np.concatenate((data['train_images'], data['test_images']), axis=0)
    all_labels = np.concatenate((data['train_labels'], data['test_labels']), axis=0)
    
    train_images, test_images, train_labels, test_labels = train_test_split(
        all_images, all_labels, test_size=CONFIG['split_ratio'], random_state=42
    )
    
    # Generate perturbation vectors for all images
    n_pixels_to_perturb = 1  # Perturb one pixel
    n_images = len(all_images)
    
    # Generate perturbation vectors considering image dimensions
    img_height, img_width = train_images.shape[1:3]
    perturbation_vectors = torch.cat([
        torch.randint(0, img_height, (n_images, n_pixels_to_perturb, 1)),  # x positions
        torch.randint(0, img_width, (n_images, n_pixels_to_perturb, 1)),   # y positions
        torch.randint(0, 256, (n_images, n_pixels_to_perturb, 3)),         # RGB values
    ], dim=2).reshape(n_images, -1)
    
    # Convert images to torch tensors
    train_images = torch.from_numpy(train_images).float()
    test_images = torch.from_numpy(test_images).float()
    
    # Apply perturbation to all images
    perturbed_train_images = []
    for i, img in enumerate(train_images):
        perturbed = perturb_image_torch(perturbation_vectors[i].unsqueeze(0), img.unsqueeze(0))
        perturbed_train_images.append(perturbed)
    train_images = torch.stack(perturbed_train_images)
    
    perturbed_test_images = []
    for i, img in enumerate(test_images):
        perturbed = perturb_image_torch(perturbation_vectors[i+len(train_images)].unsqueeze(0), img.unsqueeze(0))
        perturbed_test_images.append(perturbed)
    test_images = torch.stack(perturbed_test_images)

        # Modify these lines to fix dimensions:
    train_images = torch.stack(perturbed_train_images).squeeze(1)  # Remove extra dimension
    test_images = torch.stack(perturbed_test_images).squeeze(1)    # Remove extra dimension
    
    # For grayscale, convert RGB to single channel
    if CONFIG['num_channels'] == 1:
        train_images = train_images.mean(dim=-1, keepdim=True)
        test_images = test_images.mean(dim=-1, keepdim=True)
        train_images = train_images.permute(0, 3, 1, 2)  # [B, C, H, W]
        test_images = test_images.permute(0, 3, 1, 2)    # [B, C, H, W]
    else:
        train_images = train_images.permute(0, 3, 1, 2)  # [B, C, H, W]
        test_images = test_images.permute(0, 3, 1, 2)    # [B, C, H, W]
    
    split = int(len(train_images) * 0.9)
    large_split_images = train_images[:split]
    small_split_images = train_images[split:]
    large_split_labels = train_labels[:split]
    small_split_labels = train_labels[split:]

    data_transform = transforms.Compose([
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    def custom_transform(x):
        if isinstance(x, torch.Tensor):
            x = x.float() / 255.0
        else:
            x = torch.FloatTensor(x) / 255.0
        return data_transform(x)
        
    datasets = {
        'Full': MedMNISTDataset(train_images, train_labels, transform=custom_transform),
        'LargeSplit': MedMNISTDataset(large_split_images, large_split_labels, transform=custom_transform),
        'SmallSplit': MedMNISTDataset(small_split_images, small_split_labels, transform=custom_transform),
        'test': MedMNISTDataset(test_images, test_labels, transform=custom_transform)
    }

    loaders = {name: DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
               for name, dataset in datasets.items()}

    class_counts = {
        'Full': np.bincount(train_labels.flatten()),
        'LargeSplit': np.bincount(large_split_labels.flatten()),
        'SmallSplit': np.bincount(small_split_labels.flatten()),
        'test': np.bincount(test_labels.flatten())
    }

    return loaders, class_counts