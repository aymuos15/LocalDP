import torch

# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_dice_score(model, dataset, device, threshold=0.5):
    """
    Calculate the Dice score for a model on a given dataset.

    Parameters:
    model (torch.nn.Module): The model to evaluate.
    dataset (torch.utils.data.Dataset): The dataset to evaluate on.
    device (torch.device): The device to use for computation.
    threshold (float): The threshold to binarize the model output.

    Returns:
    float: The average Dice score over the dataset, or None if no valid samples were processed.
    """
    model.eval()  # Set the model to evaluation mode
    dice_scores = []

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for i, (input_img, target_img) in enumerate(dataset):
            
            # Move input and target images to the specified device
            input_img = input_img.to(device)
            target_img = target_img.to(device)
            
            try:
                # Forward pass: compute the model output
                output = model(input_img)
                
                # Binarize the model output using the threshold
                predicted = (output > threshold).float()
                
                # Compute the intersection and union for Dice score calculation
                intersection = (predicted * target_img).sum(dim=(1, 2, 3))
                union = predicted.sum(dim=(1, 2, 3)) + target_img.sum(dim=(1, 2, 3))
                
                # Compute the Dice score for each sample in the batch
                dice = (2.0 * intersection) / (union + 1e-7)
                
                # Extend the list of Dice scores with the current batch's scores
                dice_scores.extend(dice.cpu().tolist())
            except RuntimeError as e:
                print(f"Error processing sample {i}: {str(e)}")
                continue

    # Compute the average Dice score over all samples
    if dice_scores:
        dice_scores_tensor = torch.tensor(dice_scores)
        average_dice = torch.mean(dice_scores_tensor)
        return average_dice.item()
    else:
        print("No valid samples processed.")
        return None
