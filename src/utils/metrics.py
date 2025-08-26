import torch
import numpy as np

def threshold(pred, th=0.5): 
    return (pred > th).float()

@torch.no_grad()
def segmentation_metrics(logits, targets, eps=1e-7):
    p = torch.sigmoid(logits)
    y = targets.float()
    pred = threshold(p, 0.5)
    tp = (pred*y).sum()
    tn = ((1-pred)*(1-y)).sum()
    fp = (pred*(1-y)).sum()
    fn = ((1-pred)*y).sum()
    dice = (2*tp + eps) / (2*tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)
    acc  = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    spe  = (tn + eps) / (tn + fp + eps)
    sen  = (tp + eps) / (tp + fn + eps)
    return dict(DSC=dice.item(), mIoU=iou.item(), Acc=acc.item(), Spe=spe.item(), Sen=sen.item())

# Add the calculate_metrics function that matches the import in train.py
def calculate_metrics(preds, targets, threshold=0.5):
    """
    Calculate segmentation metrics for batch predictions
    
    Args:
        preds: torch.Tensor of shape (N, 1, H, W) with raw logits
        targets: torch.Tensor of shape (N, 1, H, W) with binary masks
        threshold: float, threshold for binarizing predictions
        
    Returns:
        dict: Dictionary containing DSC, mIoU, Acc, Spe, Sen
    """
    # Convert to numpy for batch processing
    if isinstance(preds, torch.Tensor):
        preds = torch.sigmoid(preds).cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Binarize predictions
    preds = (preds > threshold).astype(np.float32)
    targets = targets.astype(np.float32)
    
    # Initialize metrics
    dsc_list = []
    iou_list = []
    acc_list = []
    spe_list = []
    sen_list = []
    
    # Calculate metrics for each image in the batch
    for i in range(preds.shape[0]):
        pred = preds[i, 0]
        target = targets[i, 0]
        
        # Calculate confusion matrix components
        tp = np.sum(pred * target)
        tn = np.sum((1 - pred) * (1 - target))
        fp = np.sum(pred * (1 - target))
        fn = np.sum((1 - pred) * target)
        
        # Skip if there are no positive samples in both prediction and target
        if tp + fp + fn == 0:
            continue
        
        # Calculate metrics
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        dsc_list.append(dice)
        iou_list.append(iou)
        acc_list.append(acc)
        spe_list.append(spe)
        sen_list.append(sen)
    
    # Return average metrics
    return {
        'DSC': np.mean(dsc_list) if dsc_list else 0,
        'mIoU': np.mean(iou_list) if iou_list else 0,
        'Acc': np.mean(acc_list) if acc_list else 0,
        'Spe': np.mean(spe_list) if spe_list else 0,
        'Sen': np.mean(sen_list) if sen_list else 0
    }