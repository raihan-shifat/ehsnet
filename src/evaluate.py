import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader

from models.ehsnet import EHSNet
from data.isic_dataset import ISICDataset
from utils.metrics import calculate_metrics

def main(config_path, checkpoint_path, tta=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['train']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = EHSNet(
        num_classes=config['model']['num_classes'],
        pretrained=False
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = ISICDataset(
        split_file=os.path.join(config['data']['split_dir'], 'test.txt'),
        img_dir=config['data']['img_dir'],
        mask_dir=config['data']['mask_dir'],
        size=config['data']['img_size'],
        augment=False  # No augmentations for evaluation
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=True
    )
    
    # Evaluate model
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            
            if tta:
                # Test-time augmentation
                preds = []
                for angle in [0, 90, 180, 270]:
                    # Rotate image
                    rotated_images = torch.rot90(images, k=angle//90, dims=[2, 3])
                    
                    # Forward pass
                    pred = model(rotated_images)
                    pred = torch.sigmoid(pred)
                    
                    # Rotate back
                    pred = torch.rot90(pred, k=-angle//90, dims=[2, 3])
                    preds.append(pred)
                
                # Average predictions
                pred = torch.stack(preds).mean(dim=0)
            else:
                # Standard inference
                pred = model(images)
                pred = torch.sigmoid(pred)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(masks.numpy())
    
    # Concatenate all predictions and targets
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets)
    
    # Print metrics
    print("Test Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--tta', action='store_true', help='Use test-time augmentation')
    args = parser.parse_args()
    
    main(args.config, args.checkpoint, args.tta)
