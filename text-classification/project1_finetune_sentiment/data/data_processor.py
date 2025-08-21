from transformers import AutoTokenizer
from torch.utils.data import DataLoader

def data_process(dataset, model_path=None, max_length=256, 
                    train_batch_size=32, test_batch_size=64):
    """
    对数据集进行预处理和格式化
    
    参数:
        dataset: 原始数据集
        model_path: 分词器模型路径(本地或Hugging Face Hub)
        max_length: 最大序列长度
        train_batch_size: 训练集批次大小
        test_batch_size: 测试集批次大小
        
    返回:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
    """
    try:
        # 加载分词器（支持本地路径或Hub名称）
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"成功加载分词器: {model_path}")
    except Exception as e:
        print(f"加载分词器失败: {e}")
        raise
    
    # 数据预处理函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    # 应用分词处理（使用batched加速）
    try:
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        print("数据预处理完成")
    except Exception as e:
        print(f"数据预处理失败: {e}")
        raise
    
    # 数据集格式设置
    try:
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format(
            type="torch",
            columns=["input_ids", "attention_mask", "labels"]
        )
        print("数据集格式设置完成")
    except Exception as e:
        print(f"格式设置失败: {e}")
        raise
    
    # 创建数据加载器
    try:
        train_dataset = tokenized_datasets["train"]
        test_dataset = tokenized_datasets["test"]
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=train_batch_size, 
            shuffle=True,
            num_workers=3,  # 多线程加载
            pin_memory=True  # 加速GPU数据传输
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=test_batch_size,
            num_workers=3,
            pin_memory=True
        )
        
        print(f"数据加载器创建完成 (训练批次: {train_batch_size}, 测试批次: {test_batch_size})")
        return train_loader, test_loader
    except Exception as e:
        print(f"创建数据加载器失败: {e}")
        raise

























