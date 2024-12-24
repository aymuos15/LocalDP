import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MedMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label

########### NOISE #############

class BaseNoiseTransform:
    def normalize(self, x):
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        
        # Handle grayscale vs RGB
        if len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[0] == 1):
            return transforms.Normalize(mean=[0.5], std=[0.5])(x)
        else:
            return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(x)

class GaussianNoiseTransform(BaseNoiseTransform):
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
        
    def __call__(self, x):
        x = self.normalize(x)
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x + noise, -1, 1)

class SaltPepperNoiseTransform(BaseNoiseTransform):
    def __init__(self, prob=0.05):
        self.prob = prob
        
    def __call__(self, x):
        x = self.normalize(x)
        noise_mask = torch.rand_like(x)
        x[noise_mask < self.prob/2] = -1  # pepper noise
        x[noise_mask > 1 - self.prob/2] = 1  # salt noise
        return x

class PoissonNoiseTransform(BaseNoiseTransform):
    def __init__(self, lambda_param=1.0):
        self.lambda_param = lambda_param
        
    def __call__(self, x):
        x = self.normalize(x)
        # Scale to positive values for Poisson
        x_scaled = (x + 1) * 127.5  # Scale to [0, 255]
        noise = torch.poisson(self.lambda_param * torch.ones_like(x_scaled))
        x_noisy = (x_scaled + noise) / 255.0  # Scale back to [0, 1]
        return torch.clamp(x_noisy * 2 - 1, -1, 1)  # Scale to [-1, 1]

class SpeckleNoiseTransform(BaseNoiseTransform):
    def __init__(self, noise_std=0.1):
        self.noise_std = noise_std
        
    def __call__(self, x):
        x = self.normalize(x)
        noise = torch.randn_like(x) * self.noise_std
        return torch.clamp(x * (1 + noise), -1, 1)

############# one pixel perturbation #############
#? Paper: arxiv.org/abs/1710.08864
#? Code: https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/1_one-pixel-attack-cifar10.ipynb
def perturb_image_torch(xs, img):
    # Convert numpy operations to PyTorch
    if len(xs.shape) < 2:
        xs = xs.unsqueeze(0)
    
    # Create copies of the image
    imgs = img.clone()  # Just clone the input image since we're processing one at a time
    
    # Convert to integer type
    xs = xs.long()
    
    # Split x into 5-tuples
    pixels = xs.view(-1, 5)
    for pixel in pixels:
        x_pos, y_pos, r, g, b = pixel
        # Ensure indices are within bounds
        x_pos = torch.clamp(x_pos, 0, img.shape[-2]-1)
        y_pos = torch.clamp(y_pos, 0, img.shape[-1]-1)
        
        # Handle single channel images
        if img.shape[0] == 1:
            # For grayscale images, take average of r,g,b
            gray_value = ((r + g + b) / 3).long()
            imgs[0, x_pos, y_pos] = gray_value
        else:
            # For RGB images
            imgs[:, x_pos, y_pos] = torch.tensor([r, g, b])
    
    return imgs