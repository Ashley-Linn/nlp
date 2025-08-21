import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
import os

# 1. 数据加载与预处理
def load_data(data_path):
    """从本地加载数据集"""
    return load_from_disk(data_path)

def tokenize_dataset(dataset, tokenizer, max_length=256):
    """分词并格式化数据集"""
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized = dataset.map(tokenize_function, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized

# 2. 模型初始化
def init_model(model_path, num_labels=2):
    """初始化分类模型"""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 打印参数量
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数：{total/1e6:.1f}M, 可训练参数：{trainable/1e6:.1f}M")
    return model, device

# 3. 创建数据加载器
def create_loaders(train_data, test_data, batch_size=32):
    """创建训练和测试数据加载器"""
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size*2, shuffle=False, num_workers=4)
    return train_loader, test_loader

# 4. 训练函数
def train_epoch(model, device, train_loader, optimizer, scheduler, epoch, writer):
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
    return avg_loss

# 5. 评估函数 - 增强版，支持生成详细报告
def evaluate(model, device, test_loader, epoch, writer, generate_report=False):
    """评估模型性能，可选择生成详细分类报告"""
    model.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            # 计算准确率
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)
            
            # 收集预测和标签用于报告生成
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    acc = correct / total
    if writer:
        writer.add_scalar("Accuracy/test", acc, epoch)
    
    # 生成详细报告
    report = None
    conf_mat = None
    if generate_report:
        report = classification_report(all_labels, all_preds, target_names=["negative", "positive"])
        conf_mat = confusion_matrix(all_labels, all_preds)
        print("\n=== 详细分类报告 ===")
        print(report)
        print("\n=== 混淆矩阵 ===")
        print(conf_mat)
    
    return acc, report, conf_mat

# 6. 主函数
def main():
    # 配置参数
    data_path = "D:/a_lyj/projects/local_datasets/imdb"
    model_path = "D:/a_lyj/projects/local_models/bert-base-uncased"
    batch_size = 32
    epochs = 3
    lr = 2e-5
    log_dir = "runs/bert_imdb"
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir=log_dir)
    
    # 1. 加载数据
    print("加载数据集...")
    dataset = load_data(data_path)
    train_data = dataset["train"]
    test_data = dataset["test"]
    
    # 2. 初始化分词器
    print("初始化分词器...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 3. 分词处理
    print("预处理数据...")
    train_data = tokenize_dataset(train_data, tokenizer)
    test_data = tokenize_dataset(test_data, tokenizer)
    
    # 4. 创建数据加载器
    print("创建数据加载器...")
    train_loader, test_loader = create_loaders(train_data, test_data, batch_size)
    
    # 5. 初始化模型
    print("初始化模型...")
    model, device = init_model(model_path)
    
    # 6. 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=lr, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # 7. 训练循环
    best_acc = 0
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(model, device, train_loader, optimizer, scheduler, epoch, writer)
        
        # 评估（仅计算准确率）
        val_acc, _, _ = evaluate(model, device, test_loader, epoch, writer, generate_report=False)
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            model.save_pretrained("best_model")
        
        print(f"Epoch {epoch+1} | 训练损失: {train_loss:.4f} | 验证准确率: {val_acc:.4f}")
    
    # 训练完成后生成详细报告
    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")
    print("生成最终评估报告...")
    _, report, conf_mat = evaluate(model, device, test_loader, epochs, writer, generate_report=True)
    
    writer.close()

if __name__ == "__main__":
    main()