import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def evaluate(model, device, test_loader, epoch, writer=None):
    """评估模型性能"""
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
    
    acc = correct / total
    if writer:
        writer.add_scalar("Accuracy/test", acc, epoch)
    return acc

def save_best_model(model, current_acc, best_acc, save_path="../best_model"):
    """保存最佳模型"""
    if current_acc > best_acc:
        model.save_pretrained(save_path)
        return current_acc
    return best_acc