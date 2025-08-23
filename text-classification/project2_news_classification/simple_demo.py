import torch
import torch.nn as nn
import torch.optim as AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import sys, logging, os

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置随机数种子，保证结果可复现
seed_val = 42
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")


# 1. 数据预处理模块
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 对文本进行编码
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loader(df, tokenizer, max_len, batch_size):
    """创建数据加载器"""
    ds = NewsDataset(
        texts=df.content.to_numpy(),
        labels=df.category.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True
    )

# 2. 模型构建模块
class BertNewsClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BertNewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.drop = nn.Dropout(p=0.3)  # 添加dropout防止过拟合
        self.out = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        # 获取BERT的输出
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        # 添加dropout防止过拟合
        output = self.drop(pooled_output)
        # 输出层，得到分类结果
        return self.out(output)

# 3. 训练模块
def train_epoch(
    model, 
    data_loader, 
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    n_examples
):
    """训练一个epoch"""
    model = model.train()
    
    losses = []
    correct_predictions = 0
    
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, labels)
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        # 反向传播和优化
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防止梯度爆炸
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)

# 4. 评估模块
def eval_model(model, data_loader, loss_fn, device, n_examples):
    """评估模型"""
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():  # 评估时不计算梯度
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, labels)
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    
    return correct_predictions.double() / n_examples, np.mean(losses)

# 5. 主函数 - 模型训练与评估
def train_and_evaluate():
    # 配置参数
    BERT_MODEL_NAME = "bert-base-chinese"  # 使用中文BERT模型
    MAX_LEN = 256  # 文本最大长度
    BATCH_SIZE = 16
    EPOCHS = 10
    LEARNING_RATE = 2e-5  # BERT通常使用较小的学习率
    
    # 新闻类别（可根据实际情况修改）
    CATEGORY_NAMES = ['体育', '财经', '娱乐', '科技', '政治']
    NUM_CLASSES = len(CATEGORY_NAMES)
    
    # 加载数据（假设数据格式为CSV，包含content和category两列）
    # 实际使用时请替换为你的数据路径
    df = pd.read_csv("news_data.csv")
    
    # 将类别转换为数字标签
    df['category'] = df['category'].map({category: idx for idx, category in enumerate(CATEGORY_NAMES)})
    
    # 划分训练集和测试集
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=seed_val)
    
    # 初始化BERT分词器
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    # 创建数据加载器
    train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
    test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)
    
    # 初始化模型
    model = BertNewsClassifier(BERT_MODEL_NAME, NUM_CLASSES)
    model = model.to(device)
    
    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    # 优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # 训练模型
    history = {
        'train_acc': [],
        'train_loss': [],
        'test_acc': [],
        'test_loss': []
    }
    
    best_accuracy = 0
    
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        
        # 训练
        start_time = time.time()
        train_acc, train_loss = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        train_time = time.time() - start_time
        
        print(f"训练准确率: {train_acc:.4f}, 训练损失: {train_loss:.4f}, 耗时: {train_time:.2f}秒")
        
        # 评估
        test_acc, test_loss = eval_model(
            model,
            test_data_loader,
            loss_fn,
            device,
            len(df_test)
        )
        
        print(f"测试准确率: {test_acc:.4f}, 测试损失: {test_loss:.4f}")
        print()
        
        # 保存历史记录
        history['train_acc'].append(train_acc.item())
        history['train_loss'].append(train_loss)
        history['test_acc'].append(test_acc.item())
        history['test_loss'].append(test_loss)
        
        # 保存最佳模型
        if test_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_bert_news_model.bin')
            best_accuracy = test_acc
    
    return model, tokenizer, CATEGORY_NAMES

# 6. 预测函数
def predict_news_category(text, model, tokenizer, category_names, max_len=256):
    """预测新闻类别"""
    model = model.eval()
    
    # 文本编码
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        _, preds = torch.max(outputs, dim=1)
    
    return category_names[preds.item()]

# 主程序入口
if __name__ == "__main__":
    # 训练模型
    model, tokenizer, categories = train_and_evaluate()
    
    # 示例预测
    sample_texts = [
        "湖人队今天在主场以108-105击败了勇士队，詹姆斯砍下32分8篮板7助攻",
        "央行今日宣布下调金融机构存款准备金率0.5个百分点，释放长期资金约1万亿元",
        "电影《流浪地球3》定档明年春节，原班人马回归引发观众期待",
        "新一代人工智能芯片发布，运算速度较上一代提升300%",
        "全国人民代表大会常务委员会今日通过多项重要法案"
    ]
    
    print("\n示例预测结果:")
    for text in sample_texts:
        category = predict_news_category(text, model, tokenizer, categories)
        print(f"文本: {text[:50]}...")
        print(f"预测类别: {category}\n")
