from transformers import BertTokenizer
from project2_news_classification.examples.base_data_processor import BaseDataset, DataLoader


class TextClassificationDataset(BaseDataset):
    """文本分类任务数据集"""
    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        max_seq_len: int = 128,
        seed: int = 42
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        super().__init__(data_path, seed)

    def _load(self) -> None:
        """加载TSV格式文本数据（文本\t标签）"""
        texts = []
        labels = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text, label = line.split("\t")
                texts.append(text)
                labels.append(int(label))  # 假设标签是整数
        self.data = (texts, labels)

    def _preprocess(self, text: str) -> dict:
        """文本预处理：分词、转换为ID、填充/截断"""
        return self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )


# 使用示例
if __name__ == "__main__":
    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    # 构建数据集
    train_dataset = TextClassificationDataset(
        data_path="train.txt",
        tokenizer=tokenizer,
        max_seq_len=128,
        seed=42
    )
    
    # 构建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        drop_last=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 模拟训练过程
    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx+1}/{len(train_loader)}")
        print(f"特征形状: {features['input_ids'].shape}")
        print(f"标签形状: {labels.shape}")
        break  # 只打印第一个批次
