from transformers import get_linear_schedule_with_warmup 
from torch.optim import AdamW 
from sklearn import metrics  
import time
import os
import torch
import numpy as np 
from utils import get_time_dif
from torch.utils.tensorboard import SummaryWriter
import re
from utils import setup_logging


def train(model, config, train_iterator, dev_iterator, optimizer_state=None):
    """
    模型训练函数（支持断点续传+紧急保存+有限断点保留）
    :param model: 待训练的模型
    :param config: 配置对象（包含训练参数）
    :param train_iterator: 训练数据迭代器
    :param dev_iterator: 验证数据迭代器
    :param optimizer_state: 优化器状态字典（断点续传时使用）
    """
    
    # 初始化日志
    logger = setup_logging(config)
    
    model.train()
    start_time = time.time()

    # 1、优化器参数分组：区分需要权重衰减和不需要权重衰减的参数（通常偏置项(bias)和层归一化参数不使用权重衰减）
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    param_optimizer = model.named_parameters()  
    optimizer_grouped_parameters = [
        # 对需要权重衰减的参数分组
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
         "weight_decay": config.weight_decay},
        # 对不需要权重衰减的参数分组（权重衰减设为0）
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
         "weight_decay": 0.0}
    ]
    
    # 2. 初始化优化器和调度器
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    t_total = len(train_iterator) * config.num_epochs  
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=config.warmup_steps,
        num_training_steps=t_total
    )

    # 3. 断点续传：恢复训练状态
    # 初始化基础状态（首次训练）
    start_epoch = 0  # 起始轮次
    total_batch = 0  # 总批次数
    last_improve = 0  # 最后提升批次
    best_dev_loss = float("inf")  # 最佳验证损失（初始化为无穷大）

    # 如果有续传状态，加载历史数据
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state["optimizer"])
            scheduler.load_state_dict(optimizer_state["scheduler"])
            start_epoch = optimizer_state["epoch"]
            total_batch = optimizer_state["total_batch"]
            last_improve = optimizer_state["last_improve"]
            best_dev_loss = optimizer_state["best_dev_loss"]
            logger.info(f"已恢复训练状态：起始轮次 {start_epoch+1}，当前总批次数 {total_batch}")
        except RuntimeError as e:
            logger.warning(f"断点状态不兼容，忽略历史优化器状态：{str(e)}")
    
    # 4. 核心工具函数定义
    def save_checkpoint(model, optimizer, scheduler, epoch, total_batch, last_improve, 
                        best_dev_loss, config, is_emergency=False):
        """保存断点，支持紧急保存和自动清理旧断点"""
        
        # 确定保存路径
        if is_emergency:
            checkpoint_path = os.path.join(os.path.dirname(config.checkpoint_template), "emergency.pt")
        else:
            checkpoint_path = config.checkpoint_template.replace("{epoch}", str(epoch))
        
        # 保存断点内容        
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,  # 当前epoch（已完成的轮次）
            "total_batch": total_batch,
            "last_improve": last_improve,
            "best_dev_loss": best_dev_loss
        }
        
        torch.save(checkpoint, checkpoint_path) 

        log_msg = f"紧急保存断点至：{checkpoint_path}" if is_emergency else f"保存断点至：{checkpoint_path}"
        logger.info(log_msg)

        # 非紧急保存时，清理旧断点（仅保留最近5轮）
        if not is_emergency:
            ckpt_dir = os.path.dirname(config.checkpoint_template)
            
            # 获取所有正常断点（排除紧急断点）
            all_ckpts = [
                f for f in os.listdir(ckpt_dir) 
                if f.startswith("epoch_") and f.endswith(".pt")
            ]
           
            # 按epoch排序（升序）            
            epoch_pattern = re.compile(r"epoch_(\d+)\.pt")  # 匹配epoch_数字.pt
            all_ckpts.sort(key=lambda x: int(epoch_pattern.search(x).group(1)))
            
            # 删除超过保留数量的旧断点
            if len(all_ckpts) > config.keep_last_ckpts:
                for old_ckpt in all_ckpts[:-config.keep_last_ckpts]:
                    old_path = os.path.join(ckpt_dir, old_ckpt)
                    os.remove(old_path)
                    logger.info(f"清理旧断点：{old_path}")
    

    # 5. 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config.data_dir, "tensorboard"))
    emergency_ckpt_exists = False  # 标记是否生成过紧急断点

    try:
        # 6. 训练主循环
        break_flag = False  # 早停标志
        for epoch in range(start_epoch, config.num_epochs):
            print(f"\nEpoch [{epoch+1}/{config.num_epochs}]")
            for _, (batch, labels) in enumerate(train_iterator):
                # 前向传播与参数更新
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                    labels=labels
                )
                loss = outputs.loss / config.gradient_accumulation_steps
                logits = outputs.logits                
                loss.backward()
                if (total_batch + 1) % config.gradient_accumulation_steps == 0:  # 累积到指定步数再更新
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                # 每log_batch批次验证并记录指标
                if total_batch % config.log_batch == 0:
                    # 计算训练指标
                    true = labels.detach().cpu()
                    pred = torch.argmax(logits.detach(), dim=1).cpu().numpy()
                    acc = metrics.accuracy_score(true, pred)
                    
                    # 验证集评估
                    dev_acc, dev_loss = eval(model, config, dev_iterator)
                    
                    # 记录TensorBoard（位置正确，与日志同步）
                    writer.add_scalar("train/loss", loss.item(), total_batch)
                    writer.add_scalar("train/acc", acc, total_batch)
                    writer.add_scalar("val/loss", dev_loss, total_batch)
                    writer.add_scalar("val/acc", dev_acc, total_batch)
                    
                    # 保存最佳模型
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        torch.save(model.state_dict(), config.best_model_path)
                        improve = "*"
                        last_improve = total_batch
                        logger.info(f"最佳模型已更新，保存至：{config.best_model_path}")
                    else:
                        improve = ""
                    
                    # 打印日志
                    time_dif = get_time_dif(start_time)
                    current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
                    msg = 'Iter: {0:>6}, LR: {1:.6f}, Batch Train Loss: {2:>5.2}, Batch Train Acc: {3:>6.2%}, Val Loss: {4:>5.2}, Val Acc: {5:>6.2%}, Time: {6} {7}'
                    print(msg.format(
                        total_batch, current_lr, loss.item(), acc, dev_loss, dev_acc, time_dif, improve
                    ))                   
                    
                    writer.add_scalar("train/lr", current_lr, total_batch)
                    
                    model.train()

                total_batch += 1

                # 早停机制
                if total_batch - last_improve > config.require_improvement:
                    print("长时间无性能提升，自动停止训练...")
                    break_flag = True
                    break
            if break_flag:
                break
            
            # 每轮结束保存断点（会自动清理旧断点）
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                total_batch=total_batch,
                last_improve=last_improve,
                best_dev_loss=best_dev_loss,
                config=config
            )

    except (Exception, KeyboardInterrupt) as e:
        # 捕获异常时保存紧急断点
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch if 'epoch' in locals() else start_epoch,
            total_batch=total_batch,
            last_improve=last_improve,
            best_dev_loss=best_dev_loss,
            config=config,
            is_emergency=True
        )
        emergency_ckpt_exists = True
        logger.error(f"训练中断：{str(e)}，已保存紧急断点", exc_info=True)  # 记录异常堆栈
        raise  # 抛出异常便于调试

    finally:
        # 确保TensorBoard writer关闭
        writer.close()
        logger.info("TensorBoard writer已关闭")
        # 正常结束时清理紧急断点（如果存在）
        if not emergency_ckpt_exists:
            emergency_path = os.path.join(os.path.dirname(config.checkpoint_template), "emergency.pt")
            if os.path.exists(emergency_path):
                os.remove(emergency_path)
                logger.info(f"清理临时紧急断点：{emergency_path}")

    logger.info("===== 训练流程结束 =====") 
   
def eval(model, config, iterator, flag=False):
    """
    模型评估函数（在验证集或测试集上）
    :param model: 训练好的模型
    :param config: 配置对象
    :param iterator: 评估数据迭代器
    :param flag: 是否返回详细评估报告（混淆矩阵、分类报告）
    :return: 准确率、平均损失，可选返回报告和混淆矩阵
    """
    # 将模型设置为评估模式（关闭dropout等训练特有操作）
    
    logger = setup_logging(config)
    
    model.eval()

    total_loss = 0  # 总损失
    all_preds = []
    all_labels = []   
    
    with torch.no_grad():
        for batch, labels in iterator:
            # 模型前向传播
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                labels=labels
            )

            loss = outputs.loss  # 批次损失
            logits = outputs.logits  # 模型输出的logits

            total_loss += loss.item()  # 累加损失
            # 将标签和预测结果转移到CPU并转换为numpy数组
            true = labels.detach().cpu().numpy()
            pred = torch.argmax(logits.detach(), dim=1).cpu().numpy()  
            # 拼接所有标签和预测结果
            all_labels.extend(true)  
            all_preds.extend(pred)
            
    # 转换为数组计算指标
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    acc = metrics.accuracy_score(all_labels, all_preds)
    avg_loss = total_loss / len(iterator)
    logger.info(f"评估完成 - 平均损失: {avg_loss:.4f}, 准确率: {acc:.4f}")
    
    # 如果需要详细报告，计算分类报告和混淆矩阵
    if flag:
        report = metrics.classification_report(
            all_labels, all_preds, 
            target_names=config.label_list,  # 标签名称（用于报告可读性）
            digits=4  # 保留4位小数
        )
        confusion = metrics.confusion_matrix(all_labels, all_preds)  # 混淆矩阵
        return acc, avg_loss, report, confusion
    return acc, avg_loss


def test(model, config, iterator):
    """
    模型测试函数（加载最佳模型并在测试集上评估）
    :param model: 模型结构
    :param config: 配置对象
    :param iterator: 测试数据迭代器
    """

    logger = setup_logging(config) 

    # 加载训练过程中保存的最佳模型参数
    model.load_state_dict(torch.load(config.saved_model))
    logger.info(f"已加载最佳模型：{config.best_model_path}")

    start_time = time.time()  
    acc, loss, report, confusion = eval(model, config, iterator, flag=True)
    
    # 打印测试结果
    logger.info(f"Test Loss: {loss:>5.2}, Test Acc: {acc:6.2%}")
    logger.info("Precision, Recall and F1-Score...")
    logger.info(report)
    logger.info("Confusion Matrix...")
    logger.info("\n" + str(confusion))  # 换行显示混淆矩阵
    logger.info(f"Time usage: {get_time_dif(start_time)}")