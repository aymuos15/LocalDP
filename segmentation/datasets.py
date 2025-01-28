import os

from torch.utils.data import Dataset
import torch
import numpy as np

# Proper Dataset
class SegmentationDataset(Dataset):
    def __init__(self, num_samples=100, img_size=32):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a background
        # input_img = torch.rand(1, self.img_size, self.img_size)
        input_img = torch.zeros(1, self.img_size, self.img_size)

        # Create a simple shape for segmentation (e.g., a circle)
        x = torch.arange(self.img_size).unsqueeze(1).repeat(1, self.img_size).float()
        y = torch.arange(self.img_size).unsqueeze(0).repeat(self.img_size, 1).float()
        center = self.img_size // 2
        radius = torch.randint(self.img_size // 8, self.img_size // 4, (1,)).item()
        mask = ((x - center)**2 + (y - center)**2 <= radius**2).float()

        # Add the shape to the input image
        input_img[0] += mask * 0.5
        input_img = torch.clamp(input_img, 0, 1)

        # Create the target segmentation mask
        target_img = mask.unsqueeze(0)

        return input_img, target_img

# SegmentationDataset with Gaussian noise
class SegmentationGaussianDataset(Dataset):
    def __init__(self, num_samples=100, img_size=32, noise_mean=0, noise_std=0.1):
        self.num_samples = num_samples
        self.img_size = img_size
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a background with Gaussian noise
        input_img = torch.normal(
            mean=self.noise_mean,
            std=self.noise_std,
            size=(1, self.img_size, self.img_size)
        )

        # Create a simple shape for segmentation (e.g., a circle)
        x = torch.arange(self.img_size).unsqueeze(1).repeat(1, self.img_size).float()
        y = torch.arange(self.img_size).unsqueeze(0).repeat(self.img_size, 1).float()
        center = self.img_size // 2
        radius = torch.randint(self.img_size // 8, self.img_size // 4, (1,)).item()
        mask = ((x - center)**2 + (y - center)**2 <= radius**2).float()

        # Add the shape to the input image
        input_img[0] += mask * 0.5
        input_img = torch.clamp(input_img, 0, 1)

        # Create the target segmentation mask
        target_img = mask.unsqueeze(0)

        return input_img, target_img

class PerturbedSegmentationDataset(Dataset):
    def __init__(self, num_samples=100, img_size=32, perturbation_percentage=10, transform=None):
        """
        Args:
            num_samples (int): Number of samples in the dataset
            img_size (int): Size of the square images
            perturbation_percentage (int/str): Percentage of pixels to perturb (0-100) or 'one' for single pixel
            transform: Optional transform to be applied on the perturbed image
        """
        self.num_samples = num_samples
        self.img_size = img_size
        self.perturbation_percentage = perturbation_percentage
        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate a background
        # input_img = torch.rand(1, self.img_size, self.img_size)
        input_img = torch.zeros(1, self.img_size, self.img_size)

        # Create a simple shape for segmentation (e.g., a circle)
        x = torch.arange(self.img_size).unsqueeze(1).repeat(1, self.img_size).float()
        y = torch.arange(self.img_size).unsqueeze(0).repeat(self.img_size, 1).float()
        center = self.img_size // 2
        radius = torch.randint(self.img_size // 8, self.img_size // 4, (1,)).item()
        mask = ((x - center)**2 + (y - center)**2 <= radius**2).float()

        # Add the shape to the input image
        input_img[0] += mask * 0.5
        input_img = torch.clamp(input_img, 0, 1)

        # Convert to numpy for perturbation
        input_np = input_img.squeeze().numpy()
        
        # Apply perturbation
        perturbed_input = self.perturb_image_percentage(input_np, self.perturbation_percentage)
        
        # Convert back to tensor
        perturbed_input = torch.from_numpy(perturbed_input).unsqueeze(0)

        # Create the target segmentation mask
        target_img = mask.unsqueeze(0)

        if self.transform:
            perturbed_input = self.transform(perturbed_input)
            target_img = self.transform(target_img)

        return perturbed_input, target_img

    def perturb_image_percentage(self, img, percent):
        # Calculate number of pixels to perturb
        total_pixels = img.shape[0] * img.shape[1]
        
        if percent is None or percent == 0:
            num_pixels = 0
        elif percent == 'one':  # Single pixel perturbation
            num_pixels = 1
        else:
            num_pixels = int(total_pixels * percent / 100)
        
        # Create a copy of the image
        perturbed_img = img.copy()
        
        # Generate random pixel positions
        positions = np.random.choice(total_pixels, num_pixels, replace=False)
        x_positions = positions // img.shape[1]
        y_positions = positions % img.shape[1]
        
        # Generate random values for selected pixels
        values = np.random.randint(0, 2, size=num_pixels)
        
        # Modify the pixels
        for i in range(num_pixels):
            perturbed_img[x_positions[i], y_positions[i]] = values[i]
            
        return perturbed_img

# PanNuke dataset

class PanNukeDataset(Dataset):
    def __init__(self, data_dir, fold):
        """
        Args:
            data_dir (str): Base directory containing PanNuke data
            fold (int): Which fold to use (1, 2, or 3)
        """
        self.data_dir = data_dir
        self.fold = fold
        
        # Load the data for the specified fold
        images_path = os.path.join(data_dir, f"Fold {fold}", "images", f"fold{fold}", "images.npy")
        masks_path = os.path.join(data_dir, f"Fold {fold}", "masks", f"fold{fold}", "masks.npy")
        
        self.images = np.load(images_path)
        masks = np.load(masks_path)
        # Take only the last slice of the masks
        self.masks = masks[:, :, :, -1]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get image and mask
        image = self.images[idx]
        mask = self.masks[idx]

        # Convert to float32 and normalize image to [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)

        # Convert to PyTorch tensors
        image = torch.from_numpy(image).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)  # (H, W) -> (1, H, W)

        return image, mask
