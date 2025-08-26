import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter
from .metrics import segmentation_metrics

class Averager:
    def __init__(self): 
        self.v = 0
        self.n = 0
    def add(self, x, n=1): 
        self.v += x * n
        self.n += n
    def item(self): 
        return self.v / max(self.n, 1)

def fit(model, train_loader, val_loader, criterion, optimizer, device, epochs=200, 
        save_dir="runs/exp", scheduler=None, amp=False, patience=200):  # patience set to 200

    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, "best_model.pth")
    
    writer = SummaryWriter(save_dir)
    best_score = -1e9  # Initialize the best score to a very low value
    
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    
    for epoch in range(1, epochs + 1):
        model.train()
        loss_meter = Averager()
        
        # Training loop
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=amp):
                logits = model(x)
                if isinstance(logits, list):
                    main_loss = criterion(logits[0], y)
                    ds_loss = 0
                    for i in range(1, len(logits)):
                        ds_loss += criterion(logits[i], y)
                    loss = main_loss + 0.4 * ds_loss / (len(logits) - 1)
                else:
                    loss = criterion(logits, y)
            
            # Backpropagation with mixed precision scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_meter.add(loss.item(), x.size(0))
        
        # Validation loop
        model.eval()
        val_scores = {"DSC": 0, "mIoU": 0, "Acc": 0, "Spe": 0, "Sen": 0}
        total_samples = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                if isinstance(logits, list):
                    logits = logits[0]
                preds = torch.sigmoid(logits) > 0.5
                m = segmentation_metrics(preds.float(), y)
                bs = x.size(0)
                for k in val_scores:
                    val_scores[k] += m[k] * bs
                total_samples += bs
        
        # Average the validation scores
        for k in val_scores:
            val_scores[k] /= max(total_samples, 1)
        
        # Write train loss and validation scores to TensorBoard
        writer.add_scalar("loss/train", loss_meter.item(), epoch)
        for k, v in val_scores.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        
        # Calculate combined score (DSC + mIoU) for saving the best model
        score = val_scores["DSC"] + val_scores["mIoU"]
        
        # Save the best model based on the combined score
        if score > best_score:
            best_score = score
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "score": score,
            }, ckpt_path)
            print(f"Saved best model with score: {score:.4f}")
        
        # Update the scheduler (if applicable)
        if scheduler is not None:
            scheduler.step()
        
        # Print epoch summary
        print(f"Epoch {epoch:03d} | loss {loss_meter.item():.4f} | val {val_scores}")
    
    writer.close()
    return model
