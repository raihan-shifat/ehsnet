import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from models.ehsnet import EHSNet
from data.isic_dataset import ISICDataset
from utils.losses import HybridLoss
from utils.metrics import calculate_metrics
from utils.train_utils import fit

def main(config_path):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create save directory
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    
    # Set device
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = EHSNet(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    ).to(device)
    
    # Create loss function
    criterion = HybridLoss(
        dice_weight=config['loss']['dice_weight'],
        focal_weight=config['loss']['focal_weight'],
        boundary_weight=config['loss']['boundary_weight'],
        deep_supervision=config['loss']['deep_supervision']
    )
    
    # Create optimizer with proper type conversion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config['optimizer']['lr']),
        weight_decay=float(config['optimizer']['weight_decay'])
    )
    
    # Create scheduler with proper type conversion
    if config['scheduler']['name'] == 'CosineAnnealingWarmRestarts':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(config['scheduler']['T_0']),
            T_mult=int(config['scheduler']['T_mult']),
            eta_min=float(config['scheduler']['eta_min'])
        )
    else:
        scheduler = None
    
    # Construct paths properly (ensure correct split_dir)
    split_dir = "D:/Shifat Raihan/mimage/medical image/src/data/output"  # Correct split directory
    train_split_file = os.path.join(split_dir, 'train.txt')
    val_split_file = os.path.join(split_dir, 'val.txt')
    
    # Check if split files exist
    if not os.path.exists(train_split_file):
        print(f"Error: Training split file not found at {train_split_file}")
        print("Please run the prepare_split.py script first:")
        print(f"python src/data/prepare_split.py --root data_raw/ISIC17 --out {split_dir} --val_ratio 0.15 --test_ratio 0.15")
        return
    
    if not os.path.exists(val_split_file):
        print(f"Error: Validation split file not found at {val_split_file}")
        print("Please run the prepare_split.py script first:")
        print(f"python src/data/prepare_split.py --root data_raw/ISIC17 --out {split_dir} --val_ratio 0.15 --test_ratio 0.15")
        return
    
    # Create datasets and dataloaders
    train_dataset = ISICDataset(
        split_file=train_split_file,
        img_dir=config['data']['img_dir'],
        mask_dir=config['data']['mask_dir'],
        size=int(config['data']['img_size']),
        augment=True
    )
    
    val_dataset = ISICDataset(
        split_file=val_split_file,
        img_dir=config['data']['img_dir'],
        mask_dir=config['data']['mask_dir'],
        size=int(config['data']['img_size']),
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config['data']['batch_size']),
        shuffle=True,
        num_workers=int(config['data']['num_workers']),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config['data']['batch_size']),
        shuffle=False,
        num_workers=int(config['data']['num_workers']),
        pin_memory=True
    )
    
    # Train model
    model = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=int(config['train']['epochs']),
        save_dir=config['train']['save_dir'],
        amp=config['train']['amp']
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)
