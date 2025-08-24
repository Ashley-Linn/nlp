import torch
import random
from tqdm import tqdm
import os

class DataProcessor(object):
    """
    数据处理器类：
    - 负责加载、预处理文本数据
    - 实现迭代器，支持按批次输出模型可用的张量
    - 内聚“创建迭代器”逻辑，让数据处理闭环
    """
    def __init__(self, file_path, tokenizer, batch_size, max_seq_len, seed, device):        
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.device = device

        # 加载并预处理数据
        self.contents, self.labels = self._load_and_shuffle(file_path)
        
        # 初始化迭代器参数（批次、剩余样本标记）
        self.index = 0  # 迭代索引，记录当前处理到的批次位置
        self.residue = False  # 是否存在剩余样本（最后一个批次不足batch_size的情况）
        self.num_samples = len(self.labels)  # 总样本数（文本和标签数量一致）
        # 计算总批次数（整除部分）
        self.num_batches = self.num_samples // self.batch_size
        # 如果总样本数不能被批次大小整除，标记存在剩余样本
        if self.num_samples % self.batch_size != 0:
            self.residue = True
    

    def _load_and_shuffle(self, file_path):
        """加载数据并打乱（封装核心逻辑）"""
        contents, labels = [], []
        with open(file_path, "r", encoding="UTF-8") as f:
            for line in tqdm(f, desc="Loading data"):
                line = line.strip()
                if not line or "\t" not in line:
                    continue
                content, label = line.split("\t")
                contents.append(content)
                labels.append(label)
        
        # 固定种子打乱，保证可复现
        random.seed(self.seed)
        shuffled_idx = list(range(len(labels)))
        random.shuffle(shuffled_idx)
        return (
            [contents[i] for i in shuffled_idx], 
            [labels[i] for i in shuffled_idx]
        )
    
    def __iter__(self):
        return self
    
    def __next__(self):
        """
        迭代器方法：获取下一个批次的数据
        遵循迭代器协议，当所有批次处理完毕后抛出StopIteration
        """

         # 处理最后一个不足batch_size的剩余批次
        if self.residue and self.index == self.num_batches:
            # 截取剩余样本（从当前索引*batch_size到总样本数）
            batch_x = self.contents[self.index * self.batch_size: self.num_samples]
            batch_y = self.labels[self.index * self.batch_size: self.num_samples]
            batch = self._to_tensor(batch_x, batch_y) # 转换为张量
            self.index += 1
            return batch
        # 所有批次处理完毕，重置索引并结束迭代
        elif self.index >= self.num_batches:
            self.index = 0
            raise StopIteration
        # 处理正常批次（完整的batch_size样本）
        else:
             # 截取当前批次的样本（从index*batch_size到(index+1)*batch_size）
            batch_x = self.contents[self.index * self.batch_size: (self.index+1) * self.batch_size]
            batch_y = self.labels[self.index * self.batch_size: (self.index+1) * self.batch_size]
            batch = self._to_tensor(batch_x, batch_y)
            self.index += 1
            return batch
        
    def _to_tensor(self, batch_x, batch_y):
        """
        将文本和标签转换为模型可接受的张量格式
        :param batch_x: 一个批次的文本列表
        :param batch_y: 一个批次的标签列表
        :return: tuple(inputs, labels)，模型输入张量和标签张量
        """
         # 使用分词器对文本进行编码（批量处理）
        inputs = self.tokenizer.batch_encode_plus(
            batch_x,
            padding="max_length",
            truncation="longest_first",
            max_length=self.max_seq_len,      
            return_tensors="pt"
        ) 
        batch_y = [int(y) for y in batch_y]
        labels = torch.LongTensor(batch_y).to(self.device)
        return inputs, labels  
    
    
    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches
        
    @classmethod
    def create_data_iterator(cls, 
               file_path: str, 
               tokenizer, 
               config, 
               seed: int) -> "DataProcessor":
        """
        类方法：快捷创建 DataProcessor
        作用：简化外部调用，让参数依赖更清晰
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"数据文件不存在：{file_path}")
        return cls(
            file_path=file_path,
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            seed=seed,
            device=config.device)
            


