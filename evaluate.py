import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import UNet
from src.utils import PVPanelDataset
from torch.utils.data import DataLoader

# Load the trained model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# DataLoader for testing
test_dataset = PVPanelDataset("data/val/images", "data/val/masks")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Evaluation
def evaluate():
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5  # Threshold
            
            # Visualization
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(images[0].permute(1, 2, 0).cpu().numpy())
            plt.title("Input Image")
            
            plt.subplot(1, 3, 2)
            plt.imshow(masks[0].cpu().numpy(), cmap='gray')
            plt.title("Ground Truth Mask")
            
            plt.subplot(1, 3, 3)
            plt.imshow(preds[0].cpu().numpy(), cmap='gray')
            plt.title("Predicted Mask")
            
            plt.show()

if __name__ == "__main__":
    evaluate()
