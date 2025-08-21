import os
import time 
import torch
import argparse
from tqdm import tqdm  
from train import train, test 
from config import Config
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from utils import set_seed, create_data_iterator, load_model, get_time_dif   


# 1. 解析命令行参数
# 用argparse定义可通过命令行传入的参数，让脚本更灵活
parser = argparse.ArgumentParser(description="Bert Chinese Text Classification")
parser.add_argument("--mode", type=str, required=True,  choices=["train", "eval", "test", "demo", "predict"], 
                    help="train/eval/test/demo/predict：选择运行模式（训练/评估/测试/交互演示/预测）")
parser.add_argument("--data_dir", type=str, default="./data",
                     help="训练数据、模型保存路径的根目录")
parser.add_argument("--pretrained_bert_dir", type=str, default="./pretrained_bert", 
                    help="预训练BERT模型的存放路径")
parser.add_argument("--seed", type=int, default=42, 
                    help="随机种子：保证实验可复现")
parser.add_argument("--input_file", type=str, default="./data/input.txt", 
                    help="批量预测时的输入文件路径")
parser.add_argument("--output_file", type=str, default="./data/output/output.txt", 
                    help="批量预测输出文件路径（默认./data/output.txt）")
parser.add_argument("--resume_from", type=str, default=None, 
                    help="从指定checkpoint续训（如./checkpoints/model_epoch5.pt）")
parser.add_argument("--eval_file", type=str, default="./data/dev.txt", 
                    help="独立评估模式的目标数据文件（仅mode=eval时生效）")
parser.add_argument("--save_dir", type=str, default="./saved_models", 
                    help="模型保存根目录（默认：./saved_models，替代在data目录外）")
args = parser.parse_args()  # 解析命令行传入的参数，存入args变量


# 2. 主函数：串联整个流程
def main():
    # 初始化随机种子
    set_seed(args.seed) 
    print(f"随机种子已固定：{args.seed}（保证实验可复现）")
    
    # 加载配置类
    config = Config(data_dir=args.data_dir, save_dir=args.save_dir) 
    print(f"配置加载完成，数据根目录：{args.data_dir}，模型保存路径：{args.save_dir}")

    #  加载BERT相关组件（分词器、模型配置、模型本体）    
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)  
    bert_config = BertConfig.from_pretrained(args.pretrained_bert_dir, num_labels=config.num_labels)
    model = BertForSequenceClassification.from_pretrained(os.path.join(args.pretrained_bert_dir),
                                                          config=bert_config).to(config.device)

    # 3. 根据不同模式（mode）执行逻辑
    # 3.1 训练模式（支持从checkpoint续训 + 保留训练-评估闭环）
    if args.mode == "train":
        # 记录开始时间
        start_time = time.time()  
        
        # 初始化三大数据集迭代器
        train_iterator = create_data_iterator(config.train_file, config, tokenizer, args.seed)
        dev_iterator = create_data_iterator(config.dev_file, config, tokenizer, args.seed)
        test_iterator = create_data_iterator(config.test_file, config, tokenizer, args.seed)  
        print(f"初始化数据迭代器耗时: {get_time_dif(start_time)}")
        print(f"训练集批次数量：{len(train_iterator)}，验证集批次数量：{len(dev_iterator)}，测试集批次数量：{len(test_iterator)}")

        # 从checkpoint续训（加载模型权重+优化器状态）
        resume_path = args.resume_from
        if resume_path and os.path.exists(resume_path):
            print(f"从checkpoint续训：{resume_path}")
            checkpoint = torch.load(resume_path, map_location=config.device)
            model.load_state_dict(checkpoint["model_state_dict"])  # 加载模型权重
            # 准备续传状态字典（包含优化器、调度器、进度等）
            optimizer_state = {
                "optimizer": checkpoint["optimizer"],
                "scheduler": checkpoint["scheduler"],
                "epoch": checkpoint["epoch"],
                "total_batch": checkpoint["total_batch"],
                "last_improve": checkpoint["last_improve"],
                "best_dev_loss": checkpoint["best_dev_loss"]
            }
        else:
            print("从头训练（使用预训练模型初始化）")
            optimizer_state = None  # 从头初始化优化器

        # 启动训练（传入optimizer_state用于续训）
        print("\n===== 开始训练 =====")
        train(model, config, train_iterator, dev_iterator, optimizer_state=optimizer_state)  
        
        # 保留训练-评估-测试闭环：训练结束后自动评估测试集
        print("===== 训练完成，开始测试集最终评估 =====")
        test(model, config, test_iterator) 
        print("训练+评估+测试流程全部完成！")

    elif args.mode == "eval":  
        # 3.2 独立评估模式（支持任意数据集）
        # 检查评估文件是否存在
        if not os.path.exists(args.eval_file):
            raise FileNotFoundError(f"评估文件不存在：{args.eval_file}")

        # 加载模型（使用训练好的最佳模型）
        model = load_model(model, config)
        # 初始化评估数据集迭代器
        eval_iterator = create_data_iterator(args.eval_file, config, tokenizer, args.seed)
        # 调用通用test函数进行评估（输出指标与test模式一致）
        print(f"===== 开始独立评估（文件：{args.eval_file}） =====")
        test(model, config, eval_iterator)   
    
    elif args.mode == "test": 
        # 3.3 测试模式 
        test_iterator = create_data_iterator(config.test_file, config, tokenizer, args.seed)
        model = load_model(model, config)
        print("\n===== 开始测试集评估 =====")
        test(model, config, test_iterator)
    
    elif args.mode == "demo":  
        # 3.4 交互模式
        # 加载训练好的模型权重（继续用之前训练保存的模型）
        model = load_model(model, config)
        print("交互演示模式（输入q退出）")
        # 循环接收用户输入
        while True:
            sentence = input("\n请输入文本：\n").strip()
            if sentence.lower() == "q":
                print("退出演示模式")
                break  # 直接退出循环
            if not sentence:
                print("输入不能为空，请重新输入")
                continue
            
            # 用分词器处理文本：转成模型需要的张量格式
            inputs = tokenizer(
                sentence, 
                max_length=config.max_seq_len,  # 截断/填充到max_seq_len
                truncation="longest_first",  # 超长时截断最长的序列
                return_tensors="pt"  # 返回PyTorch张量
            )
            inputs = inputs.to(config.device)  # 放到指定设备
            
            # 模型推理（前向传播）
            with torch.no_grad():  # 关闭梯度计算
                logits = model(**inputs).logits
                pred_idx = torch.argmax(logits, dim=1).item()
            print(f"分类结果：{config.label_list[pred_idx]}")        
         
    elif args.mode == "predict":    
        # 3.5 预测模式

        print("\n===== 启动批量预测模式 =====")
        model = load_model(model, config)

        # 读取待预测的文本
        if not os.path.exists(args.input_file):
                raise FileNotFoundError(f"输入文件不存在：{args.input_file}")
            
        with open(args.input_file, "r", encoding="UTF-8") as f:
            texts = [line.strip() for line in tqdm(f, desc="读取输入文件") if line.strip()]
        
        if not texts:
            print("输入文件为空，无需预测")
            return

        # 计算批次信息
        num_batches = (len(texts) - 1) // config.batch_size + 1
        results = []

        # 按批次处理文本
        for i in tqdm(range(num_batches), desc="批量预测中"):
            start = i * config.batch_size
            end = min((i + 1) * config.batch_size, len(texts))
            batch_texts = texts[start:end]
            
            # 文本编码
            inputs = tokenizer.batch_encode_plus(
                batch_texts,
                padding=True,
                max_length=config.max_seq_len,
                truncation="longest_first",
                return_tensors="pt"
            ).to(config.device)
            
            # 推理
            with torch.no_grad():
                logits = model(** inputs).logits
                pred_ids = torch.argmax(logits, dim=1).tolist()
                preds = [config.label_list[idx] for idx in pred_ids]        
            
            # 保存结果（文本+预测标签）
            results.extend([f"{text}\t{pred}" for text, pred in zip(batch_texts, preds)])

        # 输出到文件        
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True) # 确保输出目录存在
        with open(args.output_file, "w", encoding="UTF-8") as f:
            f.write("\n".join(results))
        print(f"批量预测完成，结果已保存至：{args.output_file}")

   

if __name__ == "__main__":
    main()