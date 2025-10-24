#!/usr/bin/env python3
"""
改进的训练脚本
使用新的配置系统，支持日志、混合精度训练、更好的错误处理
"""

import os
import sys
import logging
from datetime import datetime
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import pandas as pd

# 导入我们的模块
from config import Config, create_parser, parse_args_to_config
from baseline import BaselineCLIPScore, BaselineDataset, collate_fn, CLIPProcessor
from torchvision import transforms


def setup_logging(config: Config) -> logging.Logger:
    """设置日志系统"""
    # 创建日志目录
    os.makedirs(config.training.log_dir, exist_ok=True)
    
    # 创建日志文件名（带时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        config.training.log_dir,
        f"{config.experiment_name}_{timestamp}.log"
    )
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    
    return logger


def train_epoch(model, dl, opt, sched, config, scaler, logger, epoch):
    """训练一个 epoch（支持混合精度）"""
    model.train()
    totals = [0.0, 0.0, 0.0, 0.0]  # total, lq, lc, le
    
    pbar = tqdm(dl, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # 解包数据
        px, ids, mask, q_t, c_t, exp_ids, exp_mask = batch
        px = px.to(config.training.device)
        ids = ids.to(config.training.device)
        mask = mask.to(config.training.device)
        q_t = q_t.to(config.training.device)
        c_t = c_t.to(config.training.device)
        
        if exp_ids is not None:
            exp_ids = exp_ids.to(config.training.device)
            exp_mask = exp_mask.to(config.training.device)
        
        opt.zero_grad()
        
        # 混合精度训练
        try:
            if config.training.mixed_precision:
                with autocast():
                    q_p, c_p, img_g, txt_g, c_coarse = model(px, ids, mask)
                    lq = F.mse_loss(q_p, q_t)
                    lc = F.mse_loss(c_p, c_t)
                    loss = config.training.w_q * lq + config.training.w_c * lc
                    
                    le = torch.tensor(0.0, device=config.training.device)
                    if config.model.use_explanations and exp_ids is not None:
                        le = model.compute_rationale_alignment_loss(img_g, txt_g, exp_ids, exp_mask)
                        loss = loss + config.training.w_exp * le
                
                # 反向传播（混合精度）
                scaler.scale(loss).backward()
                
                # 梯度裁剪
                if config.training.max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                
                scaler.step(opt)
                scaler.update()
            else:
                # 常规训练
                q_p, c_p, img_g, txt_g, c_coarse = model(px, ids, mask)
                lq = F.mse_loss(q_p, q_t)
                lc = F.mse_loss(c_p, c_t)
                loss = config.training.w_q * lq + config.training.w_c * lc
                
                le = torch.tensor(0.0, device=config.training.device)
                if config.model.use_explanations and exp_ids is not None:
                    le = model.compute_rationale_alignment_loss(img_g, txt_g, exp_ids, exp_mask)
                    loss = loss + config.training.w_exp * le
                
                loss.backward()
                
                # 梯度裁剪
                if config.training.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                
                opt.step()
            
            sched.step()
            
            # 累积损失
            totals[0] += loss.item()
            totals[1] += lq.item()
            totals[2] += lc.item()
            totals[3] += le.item() if torch.is_tensor(le) else 0.0
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lq': f'{lq.item():.4f}',
                'lc': f'{lc.item():.4f}'
            })
            
            # 定期记录日志
            if (batch_idx + 1) % config.training.log_interval == 0:
                logger.info(
                    f"Epoch {epoch} | Batch {batch_idx+1}/{len(dl)} | "
                    f"Loss: {loss.item():.4f} (Q: {lq.item():.4f}, C: {lc.item():.4f})"
                )
        
        except RuntimeError as e:
            logger.error(f"训练错误: {e}")
            if "out of memory" in str(e):
                logger.error("显存不足！建议减小 batch_size")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            raise e
    
    n = len(dl)
    return [t / n for t in totals]


@torch.no_grad()
def evaluate(model, dl, config, logger):
    """评估模型"""
    model.eval()
    preds_q, tgts_q, preds_c, tgts_c = [], [], [], []
    
    try:
        for px, ids, mask, q_t, c_t, _, _ in tqdm(dl, desc="评估中"):
            px = px.to(config.training.device)
            ids = ids.to(config.training.device)
            mask = mask.to(config.training.device)
            
            q_p, c_p, _, _, _ = model(px, ids, mask)
            
            preds_q.extend(q_p.cpu().numpy().flatten())
            tgts_q.extend(q_t.numpy().flatten())
            preds_c.extend(c_p.cpu().numpy().flatten())
            tgts_c.extend(c_t.numpy().flatten())
        
        # 计算指标
        s_q = spearmanr(tgts_q, preds_q).correlation
        s_c = spearmanr(tgts_c, preds_c).correlation
        p_q = pearsonr(tgts_q, preds_q)[0]
        p_c = pearsonr(tgts_c, preds_c)[0]
        
        return s_q, p_q, s_c, p_c
    
    except Exception as e:
        logger.error(f"评估错误: {e}")
        raise e


def save_checkpoint(model, config, metrics, epoch, is_best=False):
    """保存检查点"""
    os.makedirs(config.training.save_dir, exist_ok=True)
    
    # 构建模型文件名
    if config.model.use_residual_learning:
        if config.model.partial_freeze:
            base_name = f"{config.experiment_name}_residual_partial_{config.model.freeze_layers}L"
        else:
            base_name = f"{config.experiment_name}_residual"
    elif config.model.use_refinement:
        base_name = f"{config.experiment_name}_refinement"
    else:
        base_name = f"{config.experiment_name}"
    
    # 保存最新模型
    checkpoint_path = os.path.join(
        config.training.save_dir,
        f"{base_name}_epoch{epoch}.pt"
    )
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'config': config
    }, checkpoint_path)
    
    # 如果是最佳模型，额外保存
    if is_best:
        best_path = os.path.join(
            config.training.save_dir,
            f"{base_name}_best.pt"
        )
        torch.save(model.state_dict(), best_path)
        return best_path
    
    return checkpoint_path


def main():
    # 解析命令行参数
    parser = create_parser()
    args = parser.parse_args()
    
    # 创建配置
    config = parse_args_to_config(args)
    
    # 如果指定了保存配置，保存后退出
    if args.save_config:
        config.save(args.save_config)
        print(f"✅ 配置已保存到: {args.save_config}")
        return
    
    # 验证配置
    try:
        config.validate()
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        sys.exit(1)
    
    # 设置日志
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("🚀 开始训练 - AIGC 图像质量评估模型")
    logger.info("=" * 80)
    
    # 打印配置
    config.print_summary()
    
    # 准备数据
    logger.info("📂 加载数据集...")
    try:
        df = pd.read_csv(config.data.data_csv_path)
        from sklearn.model_selection import train_test_split
        train_df, val_df = train_test_split(
            df,
            test_size=config.data.test_size,
            random_state=config.data.random_seed
        )
        
        logger.info(f"  训练集: {len(train_df)} 样本")
        logger.info(f"  验证集: {len(val_df)} 样本")
    except Exception as e:
        logger.error(f"❌ 数据加载失败: {e}")
        sys.exit(1)
    
    # 创建数据集和加载器
    logger.info("🔧 准备数据加载器...")
    proc = CLIPProcessor.from_pretrained(config.data.clip_model_name)
    
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(config.data.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            proc.image_processor.image_mean,
            proc.image_processor.image_std
        )
    ])
    
    tf_val = transforms.Compose([
        transforms.Resize((config.data.image_size, config.data.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            proc.image_processor.image_mean,
            proc.image_processor.image_std
        )
    ])
    
    train_ds = BaselineDataset(train_df, config.data.image_base_dir, proc, tf_train)
    val_ds = BaselineDataset(val_df, config.data.image_base_dir, proc, tf_val)
    
    train_dl = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=True if config.training.device == "cuda" else False
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=True if config.training.device == "cuda" else False
    )
    
    # 创建模型
    logger.info("🤖 创建模型...")
    refinement_cfg = {
        'hidden_dim': config.model.refinement_dim,
        'num_layers': config.model.refinement_layers,
        'num_heads': config.model.refinement_heads
    } if config.model.use_refinement else None
    
    model = BaselineCLIPScore(
        config.data.clip_model_name,
        freeze=config.model.freeze_clip,
        use_refinement=config.model.use_refinement,
        refinement_cfg=refinement_cfg,
        use_two_branch=config.model.use_two_branch,
        use_residual_learning=config.model.use_residual_learning,
        residual_scale_q=config.model.residual_scale_q,
        residual_scale_c=config.model.residual_scale_c,
        partial_freeze=config.model.partial_freeze,
        freeze_layers=config.model.freeze_layers,
        dropout=config.model.dropout
    ).to(config.training.device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  总参数量: {total_params:,}")
    logger.info(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    # 创建优化器和调度器
    logger.info("⚙️  配置优化器...")
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay
    )
    
    total_steps = len(train_dl) * config.training.epochs
    warmup_steps = int(config.training.warmup_ratio * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    
    logger.info(f"  总训练步数: {total_steps}")
    logger.info(f"  Warmup 步数: {warmup_steps}")
    
    # 混合精度训练
    scaler = GradScaler() if config.training.mixed_precision else None
    if config.training.mixed_precision:
        logger.info("  ⚡ 混合精度训练已启用")
    
    # 训练循环
    logger.info("\n" + "=" * 80)
    logger.info("🎯 开始训练")
    logger.info("=" * 80 + "\n")
    
    best_sc = -1
    best_epoch = 0
    
    try:
        for epoch in range(1, config.training.epochs + 1):
            # 训练
            train_loss, lq, lc, le = train_epoch(
                model, train_dl, opt, sched, config, scaler, logger, epoch
            )
            
            # 评估
            s_q, p_q, s_c, p_c = evaluate(model, val_dl, config, logger)
            
            # 记录日志
            log_msg = (
                f"Epoch {epoch}/{config.training.epochs} | "
                f"Train Loss: {train_loss:.4f} (Q: {lq:.4f}, C: {lc:.4f}) | "
                f"Val SROCC: Q={s_q:.4f}, C={s_c:.4f} | "
                f"Val PLCC: Q={p_q:.4f}, C={p_c:.4f}"
            )
            logger.info(log_msg)
            
            # 保存检查点
            metrics = {
                'train_loss': train_loss,
                'val_srocc_q': s_q,
                'val_srocc_c': s_c,
                'val_plcc_q': p_q,
                'val_plcc_c': p_c
            }
            
            is_best = s_c > best_sc
            if is_best:
                best_sc = s_c
                best_epoch = epoch
            
            if config.training.save_best_only:
                if is_best:
                    save_path = save_checkpoint(model, config, metrics, epoch, is_best=True)
                    logger.info(f"✅ 新的最佳模型! SROCC_C={s_c:.4f} -> {save_path}")
            else:
                save_path = save_checkpoint(model, config, metrics, epoch, is_best=is_best)
                if is_best:
                    logger.info(f"✅ 新的最佳模型! SROCC_C={s_c:.4f}")
    
    except KeyboardInterrupt:
        logger.info("\n⚠️  训练被用户中断")
    except Exception as e:
        logger.error(f"\n❌ 训练出错: {e}")
        raise e
    
    # 训练完成
    logger.info("\n" + "=" * 80)
    logger.info("🎉 训练完成！")
    logger.info(f"  最佳 SROCC_C: {best_sc:.4f} (Epoch {best_epoch})")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
