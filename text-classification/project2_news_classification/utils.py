import os
import torch
import numpy as np 
import time
from datetime import timedelta
import logging
from logging.handlers import RotatingFileHandler 

# 全局日志对象，避免重复初始化
logger = None

def set_seed(seed):
    """固定全流程随机种子（保证实验可复现）"""
    np.random.seed(seed)  # 固定numpy的随机行为
    torch.manual_seed(seed)  # 固定PyTorch CPU的随机行为
    torch.cuda.manual_seed_all(seed)  # 固定PyTorch GPU的随机行为（多GPU场景）
    torch.backends.cudnn.deterministic = True  # 强制CuDNN使用确定性算法（避免GPU随机差异）
    torch.backends.cudnn.benchmark = False  # 关闭自动优化（可能导致随机性）


def load_model(model, config):
    """加载训练好的模型权重，带错误处理"""    
    if not os.path.exists(config.best_model_path):
        raise FileNotFoundError(f"模型文件不存在：{config.best_model_path}")
    try:    
        model.load_state_dict(torch.load(
            config.best_model_path,
            map_location=config.device)  
        )
        model.to(config.device)
        model.eval()  
        get_logger().info(f"成功加载模型：{config.best_model_path}")  
        return model
    except Exception as e:
        raise RuntimeError(f"模型加载失败：{str(e)}，请先训练模型或检查文件完整性")
    

def get_time_dif(start_time):
    """计算耗时（返回人类可读的 timedelta）"""
    end_time = time.time()
    return timedelta(seconds=int(round(end_time - start_time))) # # 四舍五入为整数秒，转换为timedelta对象（方便人类可读）


def setup_logging(log_root):
    """
    初始化日志系统（程序启动时调用 1 次即可）：
    - 输出到控制台 + 按大小轮转的文件
    - 适合记录训练过程、异常等需持久化的信息
    """
    global logger
    if logger is not None:
        return logger  # 避免重复初始化    
    
    log_dir = os.path.join(log_root, "train_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    
    # 日志格式：时间 - 级别 - 消息
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 文件处理器：单个日志最大 10MB，保留 1 个备份
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=1, encoding="utf-8"
    )
    # 控制台处理器
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


def get_logger() :
    """获取全局日志对象，确保日志唯一"""
    global logger
    if logger is None:
        # 未初始化时默认简单配置（实际应在程序启动时调用 setup_logging）
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    return logger

