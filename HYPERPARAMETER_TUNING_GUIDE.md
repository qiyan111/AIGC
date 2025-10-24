# 超参数调优完全指南

## 📋 目录
1. [现有可调参数](#现有可调参数)
2. [关键参数调优建议](#关键参数调优建议)
3. [建议新增参数](#建议新增参数)
4. [自动化超参数搜索](#自动化超参数搜索)
5. [典型配置方案](#典型配置方案)

---

## 🎛️ 现有可调参数

### 1. 基础训练参数

| 参数 | 命令行 | 默认值 | 推荐范围 | 说明 |
|------|--------|--------|----------|------|
| **epochs** | `--epochs` | 20 | 10-50 | 训练轮数，数据少时可增加 |
| **batch_size** | `--batch_size` | 32 | 8-128 | 批次大小，受显存限制 |
| **lr** | `--lr` | 3e-5 | 1e-6 ~ 1e-3 | 学习率，冻结时可更大 |
| **weight_decay** | ❌ 未暴露 | 1e-4 | 1e-5 ~ 1e-3 | L2 正则化，防过拟合 |

```bash
# 示例
python baseline.py --epochs 30 --batch_size 64 --lr 5e-5
```

---

### 2. 损失权重参数 ⭐ 重要

| 参数 | 命令行 | 默认值 | 推荐范围 | 说明 |
|------|--------|--------|----------|------|
| **w_q** | `--w_q` | 0.5 | 0.3-0.7 | Quality 损失权重 |
| **w_c** | `--w_c` | 0.5 | 0.3-0.7 | Consistency 损失权重 |
| **w_exp** | `--w_exp` | 0.1 | 0.05-0.3 | Explanation 损失权重 |

**调优建议**：
- 如果更关注 **图像质量**：`--w_q 0.7 --w_c 0.3`
- 如果更关注 **文图对齐**：`--w_q 0.3 --w_c 0.7`
- **平衡模式**（推荐）：`--w_q 0.5 --w_c 0.5`

```bash
# 示例：偏重 consistency
python baseline.py --w_q 0.3 --w_c 0.7
```

---

### 3. 残差学习参数 🔥 核心

| 参数 | 命令行 | 默认值 | 推荐范围 | 说明 |
|------|--------|--------|----------|------|
| **use_residual_learning** | `--no_residual_learning` | True | - | 禁用残差学习 |
| **residual_scale_q** | `--residual_scale_q` | 0.2 | 0.1-0.5 | Quality 残差缩放 |
| **residual_scale_c** | `--residual_scale_c` | 0.2 | 0.1-0.5 | Consistency 残差缩放 |

**调优策略**：

| 数据规模 | residual_scale_q | residual_scale_c | 原因 |
|----------|-----------------|-----------------|------|
| **<500 样本** | 0.1 | 0.1 | 数据少，强约束防过拟合 |
| **500-3K** | 0.2 | 0.2 | ✅ 默认值，平衡 |
| **3K-10K** | 0.3 | 0.3 | 数据充足，允许更大调整 |
| **>10K** | 0.4-0.5 | 0.4-0.5 | 大数据，可激进 |

```bash
# 小数据集（保守）
python baseline.py --residual_scale_q 0.1 --residual_scale_c 0.1

# 大数据集（激进）
python baseline.py --residual_scale_q 0.4 --residual_scale_c 0.4
```

---

### 4. CLIP 冻结策略 ❄️ 重要

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| **freeze_clip** | `--freeze_clip` | False | 完全冻结 CLIP |
| **partial_freeze** | `--partial_freeze` | False | 部分冻结（推荐）|
| **freeze_layers** | `--freeze_layers` | 8 | 冻结前 N 层 |

**不同模型的层数配置**：

| CLIP 模型 | 总层数 | 推荐 freeze_layers | 可训练层 |
|-----------|--------|-------------------|----------|
| **ViT-B/32** | 12 | 8-10 | 2-4 层 |
| **ViT-B/16** | 12 | 8-10 | 2-4 层 |
| **ViT-L/14** | 24 | 18-20 | 4-6 层 |
| **ViT-L/14@336** | 24 | 20-22 | 2-4 层 |

**冻结策略对比**：

```bash
# 策略 1: 完全微调（数据充足，>5K）
python baseline.py
# 学习率: 1e-5 ~ 3e-5

# 策略 2: 部分冻结（推荐，数据中等 1K-5K）⭐
python baseline.py --partial_freeze --freeze_layers 18
# 学习率: 3e-5 ~ 1e-4

# 策略 3: 完全冻结（数据极少，<500）
python baseline.py --freeze_clip
# 学习率: 1e-3 ~ 5e-3（可以用大学习率）
```

---

### 5. Refinement Module 参数（旧方案，不推荐与残差学习同用）

| 参数 | 命令行 | 默认值 | 说明 |
|------|--------|--------|------|
| **use_refinement** | `--use_refinement` | False | 启用精细化模块 |
| **refinement_layers** | `--refinement_layers` | 4 | Transformer 层数 |
| **refinement_heads** | `--refinement_heads` | 8 | 注意力头数 |
| **refinement_dim** | `--refinement_dim` | 256 | 隐藏层维度 |

**注意**：与残差学习互斥，建议使用残差学习。

---

### 6. 数据增强参数

| 参数 | 代码位置 | 默认值 | 可调范围 |
|------|----------|--------|----------|
| **image_size** | `TrainingConfig` | 224 | 224-336 |
| **RandomResizedCrop scale** | 代码中 | (0.8, 1.0) | (0.6, 1.0) |
| **test_size** | ❌ 未暴露 | 0.2 | 0.1-0.3 |

---

## 🎯 关键参数调优建议

### 场景 1: 数据量 < 1K（小数据集）

```bash
python baseline.py \
    --freeze_clip \
    --residual_scale_q 0.1 \
    --residual_scale_c 0.1 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-3 \
    --w_q 0.5 \
    --w_c 0.5
```

**策略**：
- ❄️ 完全冻结 CLIP（保护预训练知识）
- 📏 小残差缩放（强约束）
- 📚 更多 epochs（小数据需要多轮）
- 🚀 大学习率（只训练预测头）

---

### 场景 2: 数据量 1K-5K（中等数据集）⭐ 推荐

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
    --w_c 0.5
```

**策略**：
- 🧊 部分冻结（平衡性能和泛化）
- 📏 中等残差缩放（默认值）
- ⚖️ 平衡的损失权重

---

### 场景 3: 数据量 > 5K（大数据集）

```bash
python baseline.py \
    --residual_scale_q 0.3 \
    --residual_scale_c 0.3 \
    --epochs 15 \
    --batch_size 64 \
    --lr 3e-5 \
    --w_q 0.5 \
    --w_c 0.5
```

**策略**：
- 🔥 完全微调（数据充足）
- 📏 较大残差缩放（允许更大调整）
- 📦 大 batch size（加速训练）

---

## 💡 建议新增参数

### 1. Warmup 比例（当前硬编码）

**当前代码**：
```python
warmup_steps = int(0.05 * len(train_dl) * cfg.epochs)  # 固定 5%
```

**建议修改**：
```python
# 在 TrainingConfig 添加
self.warmup_ratio = 0.05  # 可调整为 0.01-0.1

# 在 parser 添加
parser.add_argument('--warmup_ratio', type=float, help='Warmup steps ratio')

# 在 main 中使用
warmup_steps = int(cfg.warmup_ratio * len(train_dl) * cfg.epochs)
```

**调优建议**：
- 小数据集：`0.1`（更长 warmup）
- 大数据集：`0.05`（默认）
- 完全冻结：`0.01`（几乎不需要 warmup）

---

### 2. Dropout 比例

**当前代码**：
```python
nn.Dropout(0.1)  # 固定值
```

**建议添加**：
```python
self.dropout = 0.1  # 可调整为 0.0-0.3
```

**调优建议**：
- 无过拟合：`0.0`（移除 dropout）
- 轻微过拟合：`0.1`（默认）
- 严重过拟合：`0.2-0.3`

---

### 3. 学习率调度器类型

**当前**：固定使用 `cosine_schedule_with_warmup`

**建议新增选项**：
```bash
--scheduler cosine  # 余弦退火（默认）
--scheduler linear  # 线性衰减
--scheduler step    # 阶梯衰减
--scheduler constant  # 恒定学习率
```

---

### 4. MLP 隐藏层维度

**当前代码**：
```python
self.q_delta_head = nn.Sequential(
    nn.Linear(dim, 128),  # 固定 128
    ...
)
```

**建议添加**：
```python
self.hidden_dim_q = 128  # 可调整为 64-512
self.hidden_dim_c = 256  # Consistency 的隐藏层
```

---

### 5. 梯度裁剪

**当前**：未使用梯度裁剪

**建议添加**：
```python
# 在 TrainingConfig
self.max_grad_norm = 1.0  # 0 表示不裁剪

# 在训练循环
if cfg.max_grad_norm > 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
```

**建议值**：
- 训练稳定：`0`（不裁剪）
- 梯度爆炸：`1.0`（默认）
- 严重爆炸：`0.5`

---

### 6. 早停 (Early Stopping)

**建议添加**：
```python
self.early_stopping_patience = 5  # 验证集不提升则停止
self.early_stopping_min_delta = 0.001  # 最小提升阈值
```

---

### 7. 标签平滑 (Label Smoothing)

对于回归任务可以使用软标签：
```python
self.label_smoothing = 0.0  # 0.0-0.1
```

---

## 🤖 自动化超参数搜索

### 使用 Optuna 进行自动调优

创建 `hyperparam_search_advanced.py`：

```python
import optuna

def objective(trial):
    # 搜索空间
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    residual_scale_q = trial.suggest_uniform('residual_scale_q', 0.1, 0.5)
    residual_scale_c = trial.suggest_uniform('residual_scale_c', 0.1, 0.5)
    freeze_layers = trial.suggest_int('freeze_layers', 16, 22, step=2)
    w_q = trial.suggest_uniform('w_q', 0.3, 0.7)
    w_c = 1.0 - w_q
    
    # 训练并返回验证集 SROCC
    srocc = train_with_params(lr, residual_scale_q, residual_scale_c, 
                              freeze_layers, w_q, w_c)
    return srocc

# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best SROCC: {study.best_value}")
```

**推荐搜索的参数**：
1. ⭐ `lr` (最重要)
2. ⭐ `residual_scale_q/c`
3. ⭐ `freeze_layers`
4. `w_q` / `w_c`
5. `batch_size`

---

## 📊 典型配置方案对比

### 方案 A: 快速验证（5 分钟）

```bash
python baseline.py \
    --freeze_clip \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3
```

---

### 方案 B: 平衡性能（20 分钟）⭐ 推荐

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
    --w_c 0.5
```

---

### 方案 C: 极致性能（1 小时+）

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 20 \
    --residual_scale_q 0.25 \
    --residual_scale_c 0.25 \
    --epochs 50 \
    --batch_size 16 \
    --lr 2e-5 \
    --w_q 0.4 \
    --w_c 0.6
```

---

### 方案 D: 重视 Quality

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 18 \
    --w_q 0.7 \
    --w_c 0.3 \
    --residual_scale_q 0.3 \
    --residual_scale_c 0.15
```

---

### 方案 E: 重视 Consistency

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 18 \
    --w_q 0.3 \
    --w_c 0.7 \
    --residual_scale_q 0.15 \
    --residual_scale_c 0.3
```

---

## 🔍 参数敏感度分析

根据经验，各参数对性能的影响程度：

| 参数 | 敏感度 | 影响 SROCC | 建议优先级 |
|------|--------|------------|-----------|
| **lr** | 🔥🔥🔥 很高 | ±0.05 | ⭐⭐⭐ 最优先 |
| **freeze_layers** | 🔥🔥 高 | ±0.03 | ⭐⭐⭐ 最优先 |
| **residual_scale_c** | 🔥🔥 高 | ±0.02 | ⭐⭐ 优先 |
| **w_q / w_c** | 🔥 中 | ±0.01 | ⭐⭐ 优先 |
| **residual_scale_q** | 🔥 中 | ±0.01 | ⭐ 次要 |
| **batch_size** | 🌡️ 低 | ±0.005 | ⭐ 次要 |
| **epochs** | 🌡️ 低 | - | - 够用即可 |

---

## 🎓 调参经验法则

### 1. 从保守开始
```bash
# 第一次训练：使用默认 + 部分冻结
python baseline.py --partial_freeze --freeze_layers 18
```

### 2. 观察训练曲线
- **Train loss 下降，Val loss 上升** → 过拟合
  - 增大 `freeze_layers`（如 18→20）
  - 减小 `residual_scale`（如 0.2→0.15）
  - 增大 `weight_decay`
  
- **Train & Val loss 都很高** → 欠拟合
  - 减小 `freeze_layers`（如 18→16）
  - 增大 `residual_scale`（如 0.2→0.3）
  - 增大 `lr`

- **Train loss 下降慢** → 学习率问题
  - 尝试更大 `lr`（如 3e-5 → 5e-5）

### 3. 迭代优化
```
Baseline → 调 lr → 调 freeze_layers → 调 residual_scale → 调 loss_weight
```

---

## 📝 快速检查清单

训练前检查：
- [ ] 学习率是否适配冻结策略？
  - 完全冻结: `1e-3`
  - 部分冻结: `3e-5 ~ 1e-4`
  - 完全微调: `1e-5 ~ 3e-5`
  
- [ ] 残差缩放是否适配数据量？
  - <1K: `0.1`
  - 1K-5K: `0.2`
  - >5K: `0.3+`

- [ ] Batch size 是否充分利用显存？
  - 显存充足时尽量用大 batch (32-64)

- [ ] 损失权重是否平衡？
  - 通常 `w_q = w_c = 0.5` 最稳定

---

## 🚀 高级技巧

### 1. 两阶段训练

```bash
# 阶段 1: 冻结训练（快速收敛）
python baseline.py --freeze_clip --epochs 10 --lr 1e-3

# 阶段 2: 部分解冻微调（提升性能）
python baseline.py --partial_freeze --freeze_layers 20 \
    --epochs 20 --lr 1e-5 \
    # 加载阶段 1 的权重
```

### 2. 渐进式解冻

```bash
# 从冻结 22 层开始，逐步解冻
--freeze_layers 22  # Epoch 1-10
--freeze_layers 20  # Epoch 11-20
--freeze_layers 18  # Epoch 21-30
```

### 3. 循环学习率 (Cyclic LR)

在某些情况下比 cosine 更好，可以逃离局部最优。

---

**最后更新**: 2025-10-20  
**版本**: 2.0

