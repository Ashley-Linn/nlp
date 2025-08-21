from transformers import AutoModelForSequenceClassification
import torch

def initialize_model(model_path, num_labels):
    """初始化模型并返回设备"""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数：{total_params/1e6:.1f}M, 可训练参数：{trainable_params/1e6:.1f}M")
    
    return model, device