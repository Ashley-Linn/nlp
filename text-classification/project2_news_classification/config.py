import os
import torch


class Config(object):
    """
    配置类，用于集中管理模型训练和评估所需的所有参数与路径
    采用集中配置模式，便于参数管理、修改和复用
    """
    
    def __init__(self, data_dir, save_dir) -> None:
        """
        初始化配置
        
        Args:
            data_dir: 数据根目录
            save_dir: 模型保存根目录
        """
        
        # 数据目录及文件路径配置
        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, "train.txt")
        self.dev_file = os.path.join(data_dir, "dev.txt")
        self.label_file = os.path.join(data_dir, "label.txt")
        self.test_file = os.path.join(data_dir, "test.txt")     
        
        # 验证数据文件存在性
        for file_path, file_type in [
            (self.train_file, "训练文件"),
            (self.dev_file, "验证文件"),
            (self.label_file, "标签文件"),
            (self.test_file, "测试文件")
        ]:
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"{file_type}不存在: {file_path}")

        # 模型保存、日志保存相关路径配置
        self.save_root = save_dir
        self.saved_model_dir = os.path.join(self.save_root, "model")
        self.checkpoint_dir = os.path.join(self.save_root, "checkpoints")
        self.log_root = "./logs"
        
        # 一次性创建所有需要的目录
        for dir_path in [self.save_root, self.saved_model_dir, self.checkpoint_dir, "./logs"]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 模型文件路径定义
        self.best_model_path = os.path.join(self.saved_model_dir, "best_model.pth")
        self.checkpoint_template = os.path.join(self.checkpoint_dir, "epoch_{epoch}.pt")                
        
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
  

       
       # 以下参数未使用，仅仅是放在这里参考       
       # 预训练模型相关（默认值参考）
        # self.model_name_or_path = "bert-base-chinese"  # 预训练模型路径/名称
        # self.freeze_pretrained = False  # 是否冻结预训练层
        # self.output_hidden_states = False  # 是否使用隐藏状态特征
        
        # 数据加载相关
        # self.num_workers = 1  # 数据加载线程数
        # self.pin_memory = False  # 是否锁定内存
        # self.drop_last = False  # 是否丢弃最后一个不完整批次      

        
         
