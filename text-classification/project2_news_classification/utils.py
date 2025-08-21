import os
import torch
import numpy as np 
from preprocess import DataProcessor
from datetime import timedelta
import time
import logging
from logging.handlers import RotatingFileHandler 


# 1. 固定随机种子（可复现实验结果）
def set_seed(seed):
    np.random.seed(seed)  # 固定numpy的随机行为
    torch.manual_seed(seed)  # 固定PyTorch CPU的随机行为
    torch.cuda.manual_seed_all(seed)  # 固定PyTorch GPU的随机行为（多GPU场景）
    torch.backends.cudnn.deterministic = True  # 强制CuDNN使用确定性算法（避免GPU随机差异）
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（可能导致随机性）

# 2. 工具函数：创建数据迭代器
def create_data_iterator(file_path, config, tokenizer, seed):
    """创建指定文件的数据迭代器，封装重复逻辑"""
    try:
        return DataProcessor(
            data_file=file_path,
            device=config.device,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            seed=seed
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"数据文件不存在：{file_path}，请检查路径是否正确")

# 3. 工具函数：加载模型（复用代码）
def load_model(model, config):
    """加载训练好的模型权重，带错误处理"""
    try:
        if not os.path.exists(config.saved_model):
            raise FileNotFoundError(f"模型文件不存在：{config.saved_model}")
        
        model.load_state_dict(torch.load(
            config.saved_model,
            map_location=torch.device(config.device)  # 确保设备兼容
        ))
        model.to(config.device)
        model.eval()  # 切换到评估模式
        print(f"成功加载模型：{config.saved_model}")
        return model
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)}，请先训练模型或检查文件完整性")
    

# 4. 工具函数：计算消耗时间
def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif))) # # 四舍五入为整数秒，转换为timedelta对象（方便人类可读）

# 5.日志函数
def setup_logging(config):
    """配置日志系统：同时输出到控制台和文件，支持日志轮转"""
    """从 Config 类中读取日志根目录"""
    log_root = config.log_root
    log_dir = os.path.join(log_root, "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")

    # 日志格式：时间 - 级别 - 消息
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 设置日志处理器（轮转文件+控制台）
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=1, encoding="utf-8"
    )  # 单个日志文件最大10MB，保留1个备份
    console_handler = logging.StreamHandler()

    # 设置格式
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 配置根日志
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    return logging.getLogger(__name__)

