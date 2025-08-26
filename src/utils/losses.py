import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        """
        pred: (b, 1, h, w) logits
        target: (b, 1, h, w) binary
        """
        pred = torch.sigmoid(pred)
        
        # Compute distance maps
        dist_map = self._compute_distance_map(target)
        
        # Compute boundary loss
        loss = (pred * dist_map).mean()
        
        return loss
    
    def _compute_distance_map(self, target):
        b, _, h, w = target.shape
        dist_map = torch.zeros_like(target)
        
        for i in range(b):
            # Convert to numpy and compute distance transform
            mask = target[i, 0].cpu().numpy()
            if mask.sum() > 0:
                # Distance transform for binary mask
                dist = distance_transform_edt(1 - mask)
                dist = torch.from_numpy(dist).float().to(target.device)
                dist_map[i, 0] = dist
        
        return dist_map

class HybridLoss(nn.Module):
    def __init__(self, dice_weight=0.4, focal_weight=0.4, boundary_weight=0.2, deep_supervision=True):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.boundary_weight = boundary_weight
        self.deep_supervision = deep_supervision
        
    def forward(self, pred, target):
        if self.deep_supervision and isinstance(pred, tuple):
            # Main prediction
            main_loss = self.dice_weight * self.dice_loss(pred[0], target) + \
                       self.focal_weight * self.focal_loss(pred[0], target) + \
                       self.boundary_weight * self.boundary_loss(pred[0], target)
            
            # Deep supervision losses
            ds_loss = 0
            for i in range(1, len(pred)):
                ds_loss += self.dice_weight * self.dice_loss(pred[i], target) + \
                          self.focal_weight * self.focal_loss(pred[i], target)
            
            # Weighted sum of main and deep supervision losses
            return main_loss + 0.4 * ds_loss / (len(pred) - 1)
        else:
            return self.dice_weight * self.dice_loss(pred, target) + \
                   self.focal_weight * self.focal_loss(pred, target) + \
                   self.boundary_weight * self.boundary_loss(pred, target)