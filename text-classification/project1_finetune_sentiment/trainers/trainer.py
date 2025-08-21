import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_epoch(model, device, train_loader, optimizer, scheduler, epoch, writer=None):
    """单轮训练"""
    model.train()
    total_loss = 0
    progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in progress:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})   

    
    avg_loss = total_loss / len(train_loader)
    if writer:
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.flush()  # 确保数据写入磁盘
    return avg_loss