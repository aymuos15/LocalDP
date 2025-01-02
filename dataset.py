from torch.utils.data import Dataset

import numpy as np

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
    
class PercentagePerturbedDataset(Dataset):
    def __init__(self, images, labels, perturbation_percentage, transform=None):
        """
        Args:
            images: Original image dataset
            labels: Corresponding labels
            perturbation_percentage: Percentage of pixels to perturb (0-100)
            transform: Optional transform to be applied on the perturbed image
        """
        self.images = images
        self.labels = labels
        self.perturbation_percentage = perturbation_percentage
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply percentage-wise perturbation
        perturbed_image = perturb_image_percentage(image, self.perturbation_percentage)
        
        if self.transform:
            perturbed_image = self.transform(perturbed_image)
            
        return perturbed_image, label
    
def perturb_image_percentage(img, percent):
    # Calculate number of pixels to perturb
    total_pixels = img.shape[0] * img.shape[1]
    
    if percent == 'one': #! Single pixel perturbation
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
    if img.ndim == 3:
        values = np.random.randint(0, 2, size=(num_pixels, img.shape[2]))
    else:
        values = np.random.randint(0, 2, size=num_pixels)
    
    # Modify the pixels
    for i in range(num_pixels):
        if img.ndim == 3:
            perturbed_img[x_positions[i], y_positions[i]] = values[i]
        else:
            perturbed_img[x_positions[i], y_positions[i]] = values[i]
            
    return perturbed_img

################################################################
############# 1 pixel perturbation orig paper code #############
################################################################
#? Paper: arxiv.org/abs/1710.08864
#? Code: https://github.com/Hyperparticle/one-pixel-attack-keras/blob/master/1_one-pixel-attack-cifar10.ipynb

# def perturb_image(xs, img):
#     # If this function is passed just one perturbation vector,
#     # pack it in a list to keep the computation the same
#     if xs.ndim < 2:
#         xs = np.array([xs])
    
#     # Copy the image n == len(xs) times so that we can 
#     # create n new perturbed images
#     tile = [len(xs)] + [1]*(xs.ndim+1)
#     imgs = np.tile(img, tile)
    
#     # Make sure to floor the members of xs as int types
#     xs = xs.astype(int)
    
#     height, width = img.shape[:2]
    
#     for x, img in zip(xs, imgs):
#         # Split x into an array of 5-tuples (perturbation pixels)
#         # i.e., [[x,y,r,g,b], ...] for RGB or [[x,y,value], ...] for BW
#         pixels = np.split(x, len(x) // (2 + CONFIG['num_channels']))
#         for pixel in pixels:
#             x_pos, y_pos, *values = pixel
#             # Add bounds checking
#             x_pos = np.clip(x_pos, 0, height - 1)
#             y_pos = np.clip(y_pos, 0, width - 1)
            
#             if CONFIG['num_channels'] == 1:
#                 img[x_pos, y_pos] = values[0]
#             else:
#                 img[x_pos, y_pos] = values
    
#     return imgs