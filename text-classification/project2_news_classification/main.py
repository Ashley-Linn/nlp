import os
import time 
import torch
import argparse
from tqdm import tqdm  
from train import train, test, eval 
from config import Config
from transformers import BertConfig, BertTokenizer, BertForSequenceClassification
from preprocess import DataProcessor
from utils import set_seed, load_model, get_time_dif, setup_logging, get_logger   


def main():
    # 解析命令行参数：用argparse定义可通过命令行传入的参数，让脚本更灵活
    parser = argparse.ArgumentParser(description="Bert Chinese Text Classification")
    parser.add_argument("--mode", type=str, required=True,  
                        choices=["train", "eval", "test", "demo", "predict"], 
                        help="选择运行模式：训练/评估/测试/交互演示/预测")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="数据根目录")
    parser.add_argument("--pretrained_bert_dir", type=str, default="./pretrained_bert", 
                        help="预训练BERT模型的存放路径")
    parser.add_argument("--seed", type=int, default=42, 
                        help="随机种子：保证实验可复现")
    parser.add_argument("--input_file", type=str, default="./data/input.txt", 
                        help="批量预测时的输入文件路径")
    parser.add_argument("--output_file", type=str, default="./data/output/output.txt", 
                        help="批量预测输出文件路径")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="从指定checkpoint续训(如./checkpoints/model_epoch5.pt)")
    parser.add_argument("--eval_file", type=str, default="./data/dev.txt", 
                        help="独立评估模式的目标数据文件(仅mode=eval时生效)")
    parser.add_argument("--save_dir", type=str, default="./saved_models", 
                        help="模型保存根目录")
    args = parser.parse_args()  # 解析命令行传入的参数，存入args变量


    # 加载配置类
    config = Config(data_dir=args.data_dir, save_dir=args.save_dir)  
    print(f"数据根目录：{args.data_dir}，模型保存路径：{args.save_dir}")    
    
    # 初始化日志（全局唯一）    
    setup_logging(config.log_root)
    logger = get_logger()

    # 固定随机种子
    set_seed(args.seed) 
    logger.info(f"随机种子已固定：{args.seed}")       

    #  加载BERT相关组件（分词器、模型配置、模型本体）    
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_bert_dir)  
    bert_config = BertConfig.from_pretrained(args.pretrained_bert_dir, num_labels=config.num_labels)
    model = BertForSequenceClassification.from_pretrained(args.pretrained_bert_dir,
                                                          config=bert_config
                                                          ).to(config.device)

    # 3. 根据不同模式（mode）执行逻辑
    # 3.1 训练模式（支持从checkpoint续训 + 保留训练-评估闭环）
    if args.mode == "train":        
        start_time = time.time()  
        
        # 创建数据迭代器
        train_iterator = DataProcessor.create_data_iterator(
            file_path=config.train_file,
            tokenizer=tokenizer,
            config=config,
            seed=args.seed
        )
        dev_iterator = DataProcessor.create_data_iterator(
            file_path=config.dev_file,
            tokenizer=tokenizer,
            config=config,
            seed=args.seed
        )
        test_iterator = DataProcessor.create_data_iterator(
            file_path=config.test_file,
            tokenizer=tokenizer,
            config=config,
            seed=args.seed
        )
        
        logger.info(f"初始化数据迭代器耗时: {get_time_dif(start_time)}")
        logger.info(f"训练集批次：{len(train_iterator)}，验证集批次：{len(dev_iterator)}，测试集批次：{len(test_iterator)}")

        # 从checkpoint续训（加载模型权重+优化器状态）
        if args.resume_from and os.path.exists(args.resume_from):
            logger.info(f"从checkpoint续训：{args.resume_from}")
            # 加载checkpoint并映射到目标设备
            checkpoint = torch.load(args.resume_from, map_location=config.device)    
            # 加载模型权重        
            model.load_state_dict(checkpoint["model_state_dict"])  
            model.to(config.device)
            # 传递续传状态字典（包含优化器、调度器、进度等）
            optimizer_state = {
                "optimizer": checkpoint["optimizer"],
                "scheduler": checkpoint["scheduler"],
                "epoch": checkpoint["epoch"],
                "total_batch": checkpoint["total_batch"],
                "last_improve": checkpoint["last_improve"],
                "best_dev_loss": checkpoint["best_dev_loss"]
            }
            logger.info(f"已读取断点元数据：起始轮次 {checkpoint['epoch']+1}")
        else:
            logger.info("从头训练（使用预训练模型初始化）")
            optimizer_state = None  # 从头初始化优化器

        # 启动训练(保留训练-评估-测试闭环：训练结束后自动评估测试集)
        logger.info("\n===== 开始训练 =====")
        train(model, config, train_iterator, dev_iterator, optimizer_state=optimizer_state)  
        logger.info("===== 训练完成，开始测试集最终评估 =====")
        test(model, config, test_iterator) 
        logger.info("训练+评估+测试流程全部完成！")

    elif args.mode == "eval":  
        # 3.2 独立评估模式
        if not os.path.exists(args.eval_file):
            raise FileNotFoundError(f"评估文件不存在：{args.eval_file}")

        model = load_model(model, config)        
        eval_iterator = DataProcessor.create_data_iterator(
            file_path=args.eval_file, 
            tokenizer=tokenizer,
            config=config, 
            seed=args.seed
            )
        logger.info(f"===== 开始独立评估（文件：{args.eval_file}） =====")
        eval(model, config, eval_iterator)    
    
    elif args.mode == "test": 
        # 3.3 测试模式 
        test_iterator = DataProcessor.create_data_iterator(
            file_path=config.test_file,
            tokenizer=tokenizer,
            config=config,
            seed=args.seed
        )
        model = load_model(model, config)
        logger.info("\n===== 开始测试集评估 =====")
        test(model, config, test_iterator)
    
    elif args.mode == "demo":  
        # 3.4 交互模式
        model = load_model(model, config)
        logger.info("交互演示模式（输入q退出）")
        while True:
            sentence = input("\n请输入文本：\n").strip()
            if sentence.lower() == "q":
                logger.info("退出演示模式")
                break
            if not sentence:
                print("输入不能为空，请重新输入")
                continue
            
            # 文本编码
            inputs = tokenizer(
                sentence, 
                max_length=config.max_seq_len,
                truncation="longest_first",
                return_tensors="pt"
            ).to(config.device)
            
            # 推理
            with torch.no_grad():
                logits = model(** inputs).logits
                pred_idx = torch.argmax(logits, dim=1).item()
            print(f"分类结果：{config.label_list[pred_idx]}")    
         
    elif args.mode == "predict":    
        # 3.5 预测模式
        logger.info("\n===== 启动批量预测模式 =====")
        model = load_model(model, config)

        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"输入文件不存在：{args.input_file}")
        
        with open(args.input_file, "r", encoding="UTF-8") as f:
            texts = [line.strip() for line in tqdm(f, desc="读取输入文件") if line.strip()]
        
        if not texts:
            logger.info("输入文件为空，无需预测")
            return

        # 批量处理
        num_batches = (len(texts) - 1) // config.batch_size + 1
        results = []
        for i in tqdm(range(num_batches), desc="批量预测中"):
            start = i * config.batch_size
            end = min((i + 1) * config.batch_size, len(texts))
            batch_texts = texts[start:end]
            
            inputs = tokenizer.batch_encode_plus(
                batch_texts,
                padding=True,
                max_length=config.max_seq_len,
                truncation="longest_first",
                return_tensors="pt"
            ).to(config.device)
            
            with torch.no_grad():
                logits = model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=1).tolist()
                preds = [config.label_list[idx] for idx in pred_ids]        
            
            results.extend([f"{text}\t{pred}" for text, pred in zip(batch_texts, preds)])

        # 保存结果
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="UTF-8") as f:
            f.write("\n".join(results))
        logger.info(f"批量预测完成，结果已保存至：{args.output_file}")

if __name__ == "__main__":
    main()