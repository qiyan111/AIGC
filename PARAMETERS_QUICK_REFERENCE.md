# 🎛️ 参数快速参考卡片

## 快速查找表

### 🔥 最重要的 5 个参数（优先调整）

| 参数 | 命令行 | 默认值 | 推荐范围 | 影响 SROCC |
|------|--------|--------|----------|-----------|
| **学习率** | `--lr` | 3e-5 | 1e-6 ~ 1e-3 | ±0.05 🔥🔥🔥 |
| **冻结层数** | `--freeze_layers` | 8 | 16-22 | ±0.03 🔥🔥 |
| **Consistency 残差缩放** | `--residual_scale_c` | 0.2 | 0.1-0.5 | ±0.02 🔥🔥 |
| **Consistency 权重** | `--w_c` | 0.5 | 0.3-0.7 | ±0.01 🔥 |
| **Quality 残差缩放** | `--residual_scale_q` | 0.2 | 0.1-0.5 | ±0.01 🔥 |

---

## 📊 完整参数列表

### 1️⃣ 基础训练参数

```bash
--epochs 20              # 训练轮数（10-50）
--batch_size 32          # 批次大小（8-128，受显存限制）
--lr 3e-5               # 学习率（1e-6 ~ 1e-3）
```

**快速建议**：
- 数据少：`--epochs 30`
- 显存大：`--batch_size 64`
- 完全冻结：`--lr 1e-3`（可用大学习率）
- 部分冻结：`--lr 3e-5 ~ 1e-4`
- 完全微调：`--lr 1e-5 ~ 3e-5`

---

### 2️⃣ 优化器参数（新增✨）

```bash
--weight_decay 1e-4     # L2 正则化（1e-5 ~ 1e-3）
--warmup_ratio 0.05     # Warmup 比例（0.01-0.1）
--max_grad_norm 1.0     # 梯度裁剪（0=不裁剪，0.5-2.0）
--dropout 0.1           # Dropout 比例（0.0-0.3）
--grad_accum_steps 1    # 梯度累积步数（>1 可降显存）
--no_amp                # 禁用 AMP（默认启用，如有 CUDA）
--seed 42               # 随机种子
--num_workers 4         # DataLoader 进程数
--pin_memory            # 启用 pin memory（或 --no_pin_memory 禁用）
```

**调优建议**：
- **过拟合？** 增大 `--weight_decay 5e-4` 或 `--dropout 0.2`
- **梯度爆炸？** 减小 `--max_grad_norm 0.5`
- **训练不稳定？** 增大 `--warmup_ratio 0.1`
- **不需要正则化？** 设置 `--dropout 0.0`

---

### 3️⃣ 损失权重参数

```bash
--w_q 0.5              # Quality 损失权重（0.3-0.7）
--w_c 0.5              # Consistency 损失权重（0.3-0.7）
--w_exp 0.1            # Explanation 损失权重（0.05-0.3）
```

**场景建议**：
```bash
# 场景 1: 平衡模式（推荐）
--w_q 0.5 --w_c 0.5

# 场景 2: 重视图像质量
--w_q 0.7 --w_c 0.3

# 场景 3: 重视文图对齐
--w_q 0.3 --w_c 0.7
```

---

### 4️⃣ 残差学习参数 ⭐ 核心

```bash
--no_residual_learning      # 禁用残差学习（默认启用）
--residual_scale_q 0.2      # Quality 残差缩放（0.1-0.5）
--residual_scale_c 0.2      # Consistency 残差缩放（0.1-0.5）
```

**数据量适配**：
```bash
# <500 样本（保守）
--residual_scale_q 0.1 --residual_scale_c 0.1

# 500-3K（默认）
--residual_scale_q 0.2 --residual_scale_c 0.2

# >3K（激进）
--residual_scale_q 0.3 --residual_scale_c 0.3
```

---

### 5️⃣ CLIP 冻结策略 ❄️

```bash
--freeze_clip              # 完全冻结 CLIP
--partial_freeze           # 部分冻结（推荐）
--freeze_layers 18         # 冻结前 N 层（8-22）
```

**模型配置**：
```bash
# ViT-B/32 (12 层)
--partial_freeze --freeze_layers 8    # 训练最后 4 层

# ViT-L/14 (24 层)
--partial_freeze --freeze_layers 18   # 训练最后 6 层（推荐）
--partial_freeze --freeze_layers 20   # 训练最后 4 层（保守）
```

---

### 6️⃣ 数据参数

```bash
--data_csv_path <path>      # CSV 数据路径
--image_base_dir <path>     # 图像目录
```

---

### 7️⃣ Explanation 蒸馏（可选）

```bash
--use_explanations          # 启用 explanation 蒸馏
--w_exp 0.1                # Explanation 权重
--explanation_column <name> # CSV 列名
```

---

### 8️⃣ 传统模式参数（不推荐与残差学习同用）

```bash
--use_two_branch           # 启用双分支（cos + mlp）
--use_refinement           # 启用 Refinement Module
--refinement_layers 4      # Transformer 层数
--refinement_heads 8       # 注意力头数
--refinement_dim 256       # 隐藏层维度
--scheduler cosine|linear|constant|step   # 学习率调度器
--step_lr_step_size 1      # StepLR 衰减步长（epoch）
--step_lr_gamma 0.1        # StepLR 衰减因子
--resume_from outputs/baseline_best.pt    # 从检查点恢复
--output_dir outputs       # 输出目录
--log_csv training_log.csv # 训练日志 CSV 文件名
--early_stopping_patience 5              # 早停耐心
--early_stopping_min_delta 0.0005        # 早停最小提升
--label_scale_q 5.0       # Quality 标签缩放（默认自动）
--label_scale_c 5.0       # Consistency 标签缩放（默认自动）
```

---

## 🚀 一键配置方案

### 配置 A: 快速验证（5 分钟）

```bash
python baseline.py \
    --freeze_clip \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3
```

---

### 配置 B: 推荐平衡方案 ⭐⭐⭐

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 18 \
    --residual_scale_q 0.2 \
    --residual_scale_c 0.2 \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5 \
    --w_q 0.5 \
    --w_c 0.5 \
    --dropout 0.1 \
    --max_grad_norm 1.0
```

---

### 配置 C: 小数据集（<1K 样本）

```bash
python baseline.py \
    --freeze_clip \
    --residual_scale_q 0.1 \
    --residual_scale_c 0.1 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-3 \
    --weight_decay 5e-4 \
    --dropout 0.2
```

---

### 配置 D: 大数据集（>5K 样本）

```bash
python baseline.py \
    --residual_scale_q 0.3 \
    --residual_scale_c 0.3 \
    --epochs 15 \
    --batch_size 64 \
    --lr 3e-5 \
    --dropout 0.05
```

---

### 配置 E: 防过拟合（严重过拟合时）

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 20 \
    --residual_scale_q 0.15 \
    --residual_scale_c 0.15 \
    --dropout 0.2 \
    --weight_decay 5e-4 \
    --max_grad_norm 0.5
```

---

### 配置 F: 追求极致性能

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 18 \
    --residual_scale_q 0.25 \
    --residual_scale_c 0.25 \
    --epochs 50 \
    --batch_size 16 \
    --lr 2e-5 \
    --w_q 0.4 \
    --w_c 0.6 \
    --warmup_ratio 0.1
```

---

## 🔍 问题诊断与参数调整

### 问题 1: 训练集 Loss 下降，验证集 Loss 上升（过拟合）

**解决方案（按优先级）**：
```bash
# 方案 1: 增加冻结层数
--freeze_layers 20  # 原来 18

# 方案 2: 减小残差缩放
--residual_scale_q 0.15 --residual_scale_c 0.15  # 原来 0.2

# 方案 3: 增加正则化
--dropout 0.2 --weight_decay 5e-4  # 原来 0.1, 1e-4

# 方案 4: 减小学习率
--lr 1e-5  # 原来 3e-5
```

---

### 问题 2: 训练和验证 Loss 都很高（欠拟合）

**解决方案**：
```bash
# 方案 1: 减少冻结层数
--freeze_layers 16  # 原来 18

# 方案 2: 增大残差缩放
--residual_scale_q 0.3 --residual_scale_c 0.3  # 原来 0.2

# 方案 3: 增大学习率
--lr 5e-5  # 原来 3e-5

# 方案 4: 减少正则化
--dropout 0.05 --weight_decay 1e-5  # 原来 0.1, 1e-4
```

---

### 问题 3: Loss 震荡，训练不稳定

**解决方案**：
```bash
# 方案 1: 增加 warmup
--warmup_ratio 0.1  # 原来 0.05

# 方案 2: 梯度裁剪
--max_grad_norm 0.5  # 原来 1.0

# 方案 3: 减小学习率
--lr 1e-5  # 原来 3e-5

# 方案 4: 减小 batch size
--batch_size 16  # 原来 32
```

---

### 问题 4: SROCC_C 低，但 SROCC_Q 正常

**解决方案（Consistency 预测问题）**：
```bash
# 方案 1: 增大 Consistency 权重
--w_q 0.3 --w_c 0.7  # 原来 0.5, 0.5

# 方案 2: 增大 Consistency 残差缩放
--residual_scale_c 0.3  # 原来 0.2

# 方案 3: 减少冻结（让模型学习更多）
--freeze_layers 16  # 原来 18
```

---

### 问题 5: SROCC_Q 低，但 SROCC_C 正常

**解决方案（Quality 预测问题）**：
```bash
# 方案 1: 增大 Quality 权重
--w_q 0.7 --w_c 0.3  # 原来 0.5, 0.5

# 方案 2: 增大 Quality 残差缩放
--residual_scale_q 0.3  # 原来 0.2
```

---

## 📈 参数调优流程

### 阶段 1: Baseline（使用默认参数）

```bash
python baseline.py --partial_freeze --freeze_layers 18
```

**记录**：Val SROCC_Q, SROCC_C, Train-Val Gap

---

### 阶段 2: 调整学习率（最重要）

```bash
# 尝试 3 个学习率
python baseline.py --partial_freeze --freeze_layers 18 --lr 1e-5
python baseline.py --partial_freeze --freeze_layers 18 --lr 3e-5  # baseline
python baseline.py --partial_freeze --freeze_layers 18 --lr 5e-5
```

**选择最佳学习率**

---

### 阶段 3: 调整冻结策略

```bash
# 尝试不同冻结层数
python baseline.py --partial_freeze --freeze_layers 16 --lr <best_lr>
python baseline.py --partial_freeze --freeze_layers 18 --lr <best_lr>  # baseline
python baseline.py --partial_freeze --freeze_layers 20 --lr <best_lr>
```

---

### 阶段 4: 调整残差缩放

```bash
# 微调残差缩放
python baseline.py \
    --partial_freeze --freeze_layers <best> --lr <best_lr> \
    --residual_scale_q 0.15 --residual_scale_c 0.15

python baseline.py \
    --partial_freeze --freeze_layers <best> --lr <best_lr> \
    --residual_scale_q 0.25 --residual_scale_c 0.25
```

---

### 阶段 5: 微调损失权重和正则化

```bash
# 根据 SROCC_Q vs SROCC_C 调整权重
# 如果需要防过拟合，调整 dropout 和 weight_decay
```

---

## 💡 经验法则

### 1. 参数调整顺序（重要性递减）
```
lr → freeze_layers → residual_scale → w_q/w_c → regularization
```

### 2. 学习率与冻结策略匹配
```
完全冻结（--freeze_clip）         → lr = 1e-3 ~ 5e-3
部分冻结（--freeze_layers 18-20）  → lr = 3e-5 ~ 1e-4
完全微调                          → lr = 1e-5 ~ 3e-5
```

### 3. 残差缩放与数据量匹配
```
<500 样本   → scale = 0.1
500-3K     → scale = 0.2  ⭐ 默认
3K-10K     → scale = 0.3
>10K       → scale = 0.4-0.5
```

### 4. 正则化强度调整
```
无过拟合    → dropout=0.0, weight_decay=1e-5
轻微过拟合  → dropout=0.1, weight_decay=1e-4  ⭐ 默认
严重过拟合  → dropout=0.2, weight_decay=5e-4
```

---

## 🎯 目标 SROCC 参考

| 配置 | SROCC_Q | SROCC_C | Train-Val Gap |
|------|---------|---------|---------------|
| **优秀** | >0.90 | >0.88 | <0.02 |
| **良好** | 0.87-0.90 | 0.85-0.88 | 0.02-0.03 |
| **一般** | 0.85-0.87 | 0.82-0.85 | 0.03-0.05 |
| **需改进** | <0.85 | <0.82 | >0.05 |

---

**最后更新**: 2025-10-20  
**版本**: 1.0

**快速帮助**:
- 查看详细调优指南: `HYPERPARAMETER_TUNING_GUIDE.md`
- 查看残差学习说明: `RESIDUAL_LEARNING_USAGE.md`

