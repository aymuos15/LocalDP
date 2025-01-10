import torch
from torch import optim
from torch.utils.data import random_split

from models import SmallUNet
from datasets import SegmentationDataset
from loss import DiceLoss
from trainer import train
from metrics import calculate_dice_score

# from utils import visualize_results, calculate_dice_score

# Main execution
if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 0.001

    # MedDecathalon dataset Task04_Hippocampus
    dataset = SegmentationDataset()

    # Split dataset into training and test sets
    total_size = len(dataset)
    train_size = int(0.8 * total_size)  # 80% for training
    test_size = total_size - train_size  # Remaining 20% for test

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = SmallUNet(in_channels=1, out_channels=1).to(device)

    criterion = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, criterion, optimizer, device, num_epochs)
    print("Normal Training completed!")

    # Calculate the dice score
    dice_score = calculate_dice_score(model, test_loader, device)
    print(f"Dice score: {dice_score}")
