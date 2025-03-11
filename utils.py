import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np

class PVPanelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.bmp')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('_label.bmp')])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")  # Convert to RGB for consistent channel handling
        mask = Image.open(self.mask_paths[idx]).convert("L")  # Convert to grayscale for mask
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        image = np.array(image)
        mask = np.array(mask)
        image = torch.tensor(image).float().permute(2, 0, 1) / 255.0  # Normalize image
        mask = torch.tensor(mask).float().unsqueeze(0) / 255.0  # Binary mask
        
        return image, mask
