import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


def generate_classification_report(model, device, data_loader, target_names=None):
    """生成详细的分类评估报告和混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    # 生成报告
    report = classification_report(all_labels, all_preds, target_names=target_names)
    conf_mat = confusion_matrix(all_labels, all_preds)
    
    return report, conf_mat

def print_evaluation_metrics(report, conf_mat, title="评估结果"):
    """格式化打印评估指标"""
    print(f"\n=== {title} ===")
    print(report)
    print("\n=== 混淆矩阵 ===")
    print(conf_mat)
    print("\n")

def save_classification_report(report, save_path):
    """保存分类报告到文件"""
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"分类报告已保存至: {save_path}")

def save_confusion_matrix(conf_mat, save_path_text, save_path_image=None, class_names=None):
    """保存混淆矩阵到文本文件和图像文件"""
    # 保存为文本
    np.savetxt(save_path_text, conf_mat, fmt='%d', delimiter='\t')
    print(f"混淆矩阵(文本)已保存至: {save_path_text}")
    
    # 保存为图像
    if save_path_image:
        plt.figure(figsize=(10, 7))
        plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion matrix')
        plt.colorbar()
        
        if class_names:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)
        
        # 在混淆矩阵上标注数值
        thresh = conf_mat.max() / 2.
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                plt.text(j, i, format(conf_mat[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if conf_mat[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(save_path_image)
        print(f"混淆矩阵(图像)已保存至: {save_path_image}")