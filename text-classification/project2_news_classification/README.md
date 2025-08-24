# 前注说明：

本项目代码源于该博主代码仓：[zejunwang1/bert_text_classification: 基于 BERT 模型的中文文本分类工具](https://github.com/zejunwang1/bert_text_classification/tree/main)

**在此基础上做了一些修改和完善，但是整体未有大变化，用于实践和学习**

其中，数据部分添加了一个简单的分析EDA，思路实现参考：[kangyishuai/NEWS-TEXT-CLASSIFICATION: 零基础入门NLP - 新闻文本分类 正式赛第一名方案](https://github.com/kangyishuai/NEWS-TEXT-CLASSIFICATION?tab=readme-ov-file)


# 一、项目架构  
![Alt text](image/%E9%A1%B9%E7%9B%AE%E6%9E%B6%E6%9E%84.png)

# **二、环境准备**

**1、主要依赖**

   PyTorch  2.7.0
   Python  3.12
   transformers 4.55.4   
   scikit-learn  1.7.1    
   tensorboard  2.20.0     
   scipy   1.16.1
   tokenizers  0.21.4                 
   tqdm   4.67.1   

**2、GPU（AutoDL）**

1张RTX 3080 Ti(12GB)
要求驱动：CUDA 12.8
(虚拟机：ubuntu22.04)

# 三、数据集、模型下载

数据集：[THUCNews](https://www.kaggle.com/datasets/xianhuizhang/thucnews)

google-bert/bert-base-chinese：https://huggingface.co/bert-base-chinese/tree/main


# 四、训练过程记录

**1、训练**

```
python main.py --mode train --data_dir ./data --pretrained_bert_dir ./pretrained_bert

# 可视化
tensorboard --logdir tensorboard_logs
```
![Alt text](image/%E8%AE%AD%E7%BB%83%E5%BC%80%E5%A7%8B.png)

![Alt text](image/%E6%8F%90%E5%89%8D%E8%AE%AD%E7%BB%83%E7%BB%93%E6%9D%9F.png)

![Alt text](image/%E9%AA%8C%E8%AF%81%E9%9B%86%E6%8A%A5%E5%91%8A.png)

![Alt text](image/tensorboard%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E8%BF%87%E7%A8%8B.png)

**2、交互演示**

```
python main.py --mode demo --data_dir ./data --pretrained_bert_dir ./pretrained_bert
```
![Alt text](image/%E4%BA%A4%E4%BA%92%E6%BC%94%E7%A4%BA.png)
**3、预测**

```
python main.py --mode predict --data_dir ./data --pretrained_bert_dir ./pretrained_bert --input_file ./data/input.txt
```
![Alt text](image/%E9%A2%84%E6%B5%8B.png)

![Alt text](image/%E9%A2%84%E6%B5%8B%E7%BB%93%E6%9E%9C.png)