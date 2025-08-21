import os
from data.data_loader import data_load
from data.data_processor import data_process
from models.model_factory import initialize_model
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from trainers.trainer import train_epoch
from trainers.evaluator import evaluate, save_best_model
from utils.tensorboard import create_summary_writer
from utils.report import generate_classification_report, print_evaluation_metrics


def main():      
    # 初始化TensorBoard writer
    log_dir = './runs/exp1'  # 明确指定日志目录
    os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
    writer = create_summary_writer(log_dir=log_dir)
    
    # 1. 数据加载
    dataset_name = "imdb"  # 数据集名称
    local_path = "D:/a_lyj/projects/local_datasets/imdb"  # 本地数据集路径
    data_files={
        "train": "D:/a_lyj/projects/local_datasets/imdb/train-00000-of-00001.parquet",
        "test": "D:/a_lyj/projects/local_datasets/imdb/test-00000-of-00001.parquet",
        "unsupervised": "D:/a_lyj/projects/local_datasets/imdb/unsupervised-00000-of-00001.parquet"
    }
    cache_dir = "D:/a_lyj/projects/data_cache/imdb"  # 在线加载缓存路径
    file_format = "parquet"  # 本地数据格式
        
    dataset = data_load(
        dataset_name=dataset_name,
        local_path=local_path,
        data_files=data_files,
        cache_dir=cache_dir,
        file_format=file_format
    )
    
    # 2. 数据预处理以及创建数据加载器
    model_path = "D:/a_lyj/projects/local_models/bert-base-uncased"
    train_loader, test_loader = data_process(
        dataset,
        model_path=model_path,
        max_length=256,
        train_batch_size=32,
        test_batch_size=64
    )
        
    
    # 3. 初始化模型
    model, device = initialize_model(
        model_path=model_path, num_labels=2
    )
    
    # 4. 配置优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 3
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 5. 训练循环
    best_acc = 0
    for epoch in range(epochs):
        # 训练
        train_loss = train_epoch(
            model, device, train_loader, 
            optimizer, scheduler, epoch, writer
        )
        
        # 评估
        val_acc = evaluate(model, device, test_loader, epoch, writer)
        
        # 保存最佳模型
        best_acc = save_best_model(model, val_acc, best_acc)
        
        print(f"Epoch {epoch+1} | 训练损失: {train_loss:.4f} | 验证准确率: {val_acc:.4f} | 最佳准确率: {best_acc:.4f}")
    
    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")


    #6. 训练完成后生成评估报告
    print("开始生成最终评估报告...")
    
    # 调用utils中的评估报告生成函数    
    report, conf_mat = generate_classification_report(
        model=model,
        device=device,
        data_loader=test_loader,  # 使用测试集加载器
        target_names=["negative", "positive"]  # 根据你的标签定义
    )
    
    # 打印报告
    print_evaluation_metrics(report, conf_mat, title="最终模型评估报告")
    
    # 关闭writer
    writer.close()
    

if __name__ == "__main__":
    main()



