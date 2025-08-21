import abc
import random
from typing import List, Tuple, Any, Iterator
import torch


class BaseDataset(metaclass=abc.ABCMeta):
    """
    抽象数据集基类，定义数据处理的核心接口
    所有具体数据集类需继承此类并实现抽象方法
    """
    def __init__(self, data_path: str, seed: int = 42):
        """
        初始化数据集
        :param data_path: 数据文件路径
        :param seed: 随机种子，确保打乱可复现
        """
        self.data_path = data_path
        self.seed = seed
        self.data = None  # 存储原始数据 (特征, 标签)
        self._load()  # 加载数据
        self._shuffle()  # 打乱数据

    @abc.abstractmethod
    def _load(self) -> None:
        """
        加载数据（子类必须实现）
        需将数据解析为 (特征列表, 标签列表) 格式
        """
        pass

    @abc.abstractmethod
    def _preprocess(self, sample: Any) -> Any:
        """
        单样本预处理（子类必须实现）
        :param sample: 原始样本
        :return: 预处理后的样本
        """
        pass

    def _shuffle(self) -> None:
        """打乱数据（通用实现）"""
        if self.data is None:
            raise ValueError("数据未加载，请先调用_load方法")
        
        # 数据与标签绑定打乱
        combined = list(zip(self.data[0], self.data[1]))
        random.seed(self.seed)
        random.shuffle(combined)
        self.data = (
            [item[0] for item in combined],
            [item[1] for item in combined]
        )

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """获取单个样本（预处理后）"""
        feature, label = self.data[0][idx], self.data[1][idx]
        return self._preprocess(feature), label

    def __len__(self) -> int:
        """返回样本总数"""
        return len(self.data[0])


class DataLoader:
    """
    数据加载器，负责批次处理和格式转换
    """
    def __init__(
        self,
        dataset: BaseDataset,
        batch_size: int = 32,
        shuffle: bool = False,
        drop_last: bool = False,
        device: str = "cpu"
    ):
        """
        初始化数据加载器
        :param dataset: 数据集实例（需继承BaseDataset）
        :param batch_size: 批次大小
        :param shuffle: 每个epoch是否打乱数据
        :param drop_last: 是否丢弃最后一个不完整批次
        :param device: 数据存放设备（cpu/gpu）
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        self.indexes = list(range(len(dataset)))  # 样本索引
        self._reset()  # 初始化迭代状态

    def _reset(self) -> None:
        """重置迭代器状态"""
        self.current_idx = 0
        if self.shuffle:
            # 每个epoch重新打乱索引
            random.seed(self.dataset.seed)
            random.shuffle(self.indexes)

    def _collate_fn(self, batch: List[Tuple[Any, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批次数据拼接（可被子类重写以适应不同数据类型）
        :param batch: 单个样本组成的列表
        :return: 拼接后的特征张量和标签张量
        """
        features, labels = zip(*batch)
        
        # 这里以文本数据为例，实际使用时需根据数据类型修改
        # 特征转换为张量（假设预处理后已是可转换格式）
        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)
        return features_tensor, labels_tensor

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """返回迭代器"""
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取下一个批次"""
        if self.current_idx >= len(self.indexes):
            self._reset()
            raise StopIteration

        # 计算当前批次的结束索引
        end_idx = self.current_idx + self.batch_size
        if end_idx > len(self.indexes) and self.drop_last:
            self._reset()
            raise StopIteration

        # 获取当前批次的样本索引
        batch_indexes = self.indexes[self.current_idx:end_idx]
        self.current_idx = end_idx

        # 收集样本并拼接成批次
        batch = [self.dataset[idx] for idx in batch_indexes]
        return self._collate_fn(batch)

    def __len__(self) -> int:
        """返回批次数"""
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size
