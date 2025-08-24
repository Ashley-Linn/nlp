import pandas as pd

# 定义文件路径常量
TRAIN_FILE_PATH = "data\\train.txt"
INPUT_FILE_PATH = "data\\input.txt"

def load_text_data(file_path, has_label=False):
    """
    通用文本数据加载函数
    
    参数:
        file_path: 文件路径
        has_label: 是否包含标签（True用于训练数据，False用于输入数据）
    
    返回:
        文本内容列表，若has_label为True则同时返回标签列表
    """
    contents = []
    labels = [] if has_label else None
    
    with open(file_path, "r", encoding="UTF-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            if has_label:
                # 处理带标签的数据（格式：内容\t标签）
                if "\t" not in line:
                    continue  # 跳过格式错误的行
                content, label = line.split("\t", 1)  # 使用1次分割，避免内容中含\t的问题
                contents.append(content)
                labels.append(label)
            else:
                # 处理无标签的输入数据
                contents.append(line)
    
    return (contents, labels) if has_label else contents

def analyze_text_lengths(texts, name="文本"):
    """分析文本长度并打印统计信息"""
    length_list = [len(text) for text in texts]
    length_series = pd.Series(length_list)
    
    print(f"\n{name}长度统计:")
    stats = length_series.describe()
    print(stats)  
    return stats

if __name__ == "__main__":
    # 加载并分析训练数据
    train_contents, train_labels = load_text_data(TRAIN_FILE_PATH, has_label=True)
    train_lengths = analyze_text_lengths(train_contents, "训练文本")

    # 加载并分析输入数据
    input_contents = load_text_data(INPUT_FILE_PATH, has_label=False)
    input_lengths = analyze_text_lengths(input_contents, "输入文本")