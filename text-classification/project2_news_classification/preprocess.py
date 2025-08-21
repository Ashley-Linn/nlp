import time
import torch
import random
from tqdm import tqdm

class DataProcessor(object):
    """
    数据处理器类，用于加载、预处理文本数据，并将其转换为模型可接受的批次化张量格式
    实现了迭代器接口，支持通过for循环直接获取批次数据
    """
    def __init__(self, path, device, tokenizer, batch_size, max_seq_len, seed):
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed

        self.data = self.load(path)

        self.index = 0  # 迭代索引，记录当前处理到的批次位置
        self.residue = False  # 是否存在剩余样本（最后一个批次不足batch_size的情况）
        self.num_samples = len(self.data[0])  # 总样本数（文本和标签数量一致）
        # 计算总批次数（整除部分）
        self.num_batches = self.num_samples // self.batch_size
        # 如果总样本数不能被批次大小整除，标记存在剩余样本
        if self.num_samples % self.batch_size != 0:
            self.residue = True


    def load(self, path):
        contents = []
        labels = []
        with open(path, "r", encoding="UTF-8") as f:
            for line in tqdm(f):
                line = line.strip()
                if not line:    continue
                if line.find("\t") == -1:   continue
                content, label = line.split("\t")
                contents.append(content)
                labels.append(label)
        
        # 生成样本索引并打乱（保证数据随机性，同时通过seed固定打乱方式）
        index = list(range(len(labels)))  # 0,1,2,...,n-1
        random.seed(self.seed)  # 固定随机种子，确保打乱结果可复现
        random.shuffle(index)   # 打乱索引
        # 按打乱后的索引重新排列文本和标签
        contents = [contents[_] for _ in index]
        labels = [labels[_] for _ in index]
        return contents, labels  # 返回打乱后的数据
    
    def __next__(self):
        """
        迭代器方法：获取下一个批次的数据
        遵循迭代器协议，当所有批次处理完毕后抛出StopIteration
        """

         # 处理最后一个不足batch_size的剩余批次
        if self.residue and self.index == self.num_batches:
            # 截取剩余样本（从当前索引*batch_size到总样本数）
            batch_x = self.data[0][self.index * self.batch_size: self.num_samples]
            batch_y = self.data[1][self.index * self.batch_size: self.num_samples]
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
            batch_x = self.data[0][self.index * self.batch_size: (self.index+1) * self.batch_size]
            batch_y = self.data[1][self.index * self.batch_size: (self.index+1) * self.batch_size]
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
            return_tensors="pt"
        )
        labels = torch.LongTensor(batch_y)
        return inputs, labels
    
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.residue:
            return self.num_batches + 1
        else:
            return self.num_batches
            