from torch.utils.tensorboard import SummaryWriter

def create_summary_writer(log_dir):
    """创建 TensorBoard 写入器"""
    return SummaryWriter(log_dir=log_dir)