本项目安排架构：
project/
├── main.py                  # 主程序入口
├── run.py                   # 完整脚本
├── runs/                    # 训练结果
│   └── exp1                 # 具体展示
├── data/                    # 数据处理模块
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载
│   └── data_processor.py    # 数据预处理
├── models/                  # 模型模块
│   ├── __init__.py
│   └── model_factory.py     # 模型初始化
├── trainers/                # 训练模块
│   ├── __init__.py
│   ├── trainer.py           # 训练逻辑
│   └── evaluator.py         # 评估逻辑
└── utils/                   # 工具模块
    ├── __init__.py
    └── tensorboard.py       # TensorBoard 工具


准备工作：
阅读论文bert，了解bert模型原理

一、任务简介
采用bert-base-uncased，对经典IDMB数据集进行微调训练，实现情感判断
（本质是简单的二分类任务）


二、环境搭建
<!-- 核心库安装：transformers、datasets、torch -->


三、数据集加载与预处理
方法一：直接在线加载数据集（网络问题）
方法二：先下载到本地磁盘，再从本地磁盘加载（格式问题）

细节注意：
Hugging Face 数据集格式演进：
    早期：需要完整的结构（dataset_info.json, state.json, dataset.arrow 等）
    现在：支持纯 Parquet 文件格式（更轻量、高效）
IMDB 仓库的特殊性：
    该仓库只包含 Parquet 文件（无配置文件）
    这是 Hugging Face 优化的新格式
    设计为直接通过 load_dataset() 加载，而不是 load_from_disk()

主要是对文本数据进行分词、编码等过程。

四、初始化模型
加载预训练模型，并修改分类头

五、配置训练参数和寻训练器（直接利用api）

六、开始训练、保存模型

七、训练结果

