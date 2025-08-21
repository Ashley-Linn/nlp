import os
import torch
import logging


class Config(object):
    """
    配置类，用于集中管理模型训练和评估所需的所有参数与路径
    采用集中配置模式，便于参数管理、修改和复用
    """
    def __init__(self, data_dir, save_dir) -> None:
        
        self.data_dir = data_dir
        # 根据数据根目录拼接各数据文件的完整路径
        self.train_file = os.path.join(data_dir, "train.txt")
        self.dev_file = os.path.join(data_dir, "dev.txt")
        self.label_file = os.path.join(data_dir, "label.txt")
        self.test_file = os.path.join(data_dir, "test.txt")
        # 断言检查关键数据文件是否存在且为有效文件     
        assert os.path.isfile(self.train_file), f"训练文件 {self.train_file} 不存在"
        assert os.path.isfile(self.dev_file), f"验证文件 {self.dev_file} 不存在"
        assert os.path.isfile(self.label_file), f"标签文件 {self.label_file} 不存在"
        assert os.path.isfile(self.test_file), f"标签文件 {self.test_file} 不存在"

        # 模型保存相关路径配置
        self.save_root = save_dir         
        # 模型保存基础目录
        self.saved_model_dir = os.path.join(self.save_root, "model")
        self.checkpoint_dir = os.path.join(self.save_root, "checkpoints")
        # 最佳模型路径（固定路径，保存验证集最优模型）
        self.best_model_path = os.path.join(self.saved_model_dir, "best_model.pth")        
        # 断点续传路径模板（含 {epoch}，训练时替换）
        self.checkpoint_template = os.path.join(self.checkpoint_dir, "epoch_{epoch}.pt")        
        # 确保目录存在
        os.makedirs(self.saved_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)       
        
        # 日志保存目录
        self.log_root = "./logs"  

        # 标签配置  
        self.label_list = [label.strip() for label in open(self.label_file, "r", encoding="UTF-8").readlines()]
        self.num_labels = len(self.label_list)             
        

        # 训练控制参数       
        self.num_epochs = 3
        self.max_seq_len = 32
        self.batch_size = 128        
        self.log_batch = 100 # 每多少批次打印一次日志        
        self.require_improvement = 1000 # 多少步未提升则停止训练（早停机制）
        self.keep_last_ckpts=5

        
        # 优化器参数
        self.warmup_steps = 0
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0 # 梯度裁剪阈值（防止梯度爆炸）
        self.learning_rate = 5e-5
        self.gradient_accumulation_steps = 1  # 梯度累积步数

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"使用设备: {self.device}")  # 记录设备信息
        

       
       # 以下参数未使用，仅仅是放在这里参考       
        # # 预训练模型相关
        # self.model_name_or_path = "bert-base-chinese"  # 预训练模型路径/名称
        # self.freeze_pretrained = False  # 是否冻结预训练层
        # self.output_hidden_states = False  # 是否使用隐藏状态特征
        
        # # 数据加载相关
        # self.num_workers = 1 
        # self.pin_memory = False  
        # self.drop_last = False         
