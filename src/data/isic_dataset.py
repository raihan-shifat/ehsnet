import os
import json
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path

class ISICDataset(Dataset):
    def __init__(self, split_file, img_dir, mask_dir, size=256, augment=False):
        # Read image names from split file
        with open(split_file, 'r') as f:
            self.img_names = f.read().splitlines()
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        
        if augment:
            self.tf = A.Compose([
                A.LongestMaxSize(size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.ColorJitter(0.1,0.1,0.1,0.05, p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.tf = A.Compose([
                A.LongestMaxSize(size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_REFLECT_101),
                A.Normalize(),
                ToTensorV2()
            ])
    
    def __len__(self): 
        return len(self.img_names)
    
    def _mask_path(self, img_name):
        # Try different mask naming conventions
        candidates = [
            f"{img_name}_segmentation.png",
            f"{img_name}_segmentation.jpg", 
            f"{img_name}.png"
        ]
        
        for candidate in candidates:
            mask_path = os.path.join(self.mask_dir, candidate)
            if os.path.exists(mask_path):
                return mask_path
                
        raise FileNotFoundError(f"Mask for {img_name} not found under {self.mask_dir}")
    
    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        
        # Construct image path
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        
        # Get mask path
        mask_path = self._mask_path(img_name)
        
        # Load image and mask
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype('float32')
        
        # Apply augmentations
        augmented = self.tf(image=img, mask=mask)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0)  # Add channel dimension
        
        return img_tensor, mask_tensor