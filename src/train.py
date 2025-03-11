import sys
sys.path.append('C:\\Users\\Phiniqs\\Desktop\\PV_Solar_Segmentation\\src')  # Add the 'src' folder to the path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from .model import UNet
from .utils import PVPanelDataset

# DataLoader
train_dataset = PVPanelDataset("data/train/images", "data/train/masks")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model
model = UNet(in_channels=3, out_channels=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Loss and Optimizer
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
def train():
    for epoch in range(10):  # Change epochs as needed
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(train_loader)}")
    
    # Save the trained model
    torch.save(model.state_dict(), "model.pth")
    
if __name__ == "__main__":
    train()
