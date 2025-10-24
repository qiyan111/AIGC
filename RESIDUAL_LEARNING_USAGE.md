# 残差学习模式使用指南

## 🎯 核心原理

基于 **残差学习 + 冻结策略** 来保留 CLIP 的原始对齐空间，防止模型过拟合和偏离预训练知识。

### 核心公式

```python
# Quality 预测
q = q_base + Δq × scale_q
其中: q_base = sigmoid(Linear(CLIP_img_features))
     Δq = Tanh(MLP(CLIP_img_features))  # 范围 [-1, 1]
     scale_q = 0.2  # 限制偏离幅度

# Consistency 预测
c = cos(img, txt) + Δc × scale_c
其中: cos(img, txt) = CLIP 原始余弦相似度（映射到 [0,1]）
     Δc = Tanh(FusionHead([img, txt, cos]))  # 范围 [-1, 1]
     scale_c = 0.2  # 限制偏离幅度
```

### 三大优势

1. **保留原始对齐空间**: 基于 CLIP 的 cosine similarity 作为基准，不会完全重建表示空间
2. **限制学习范围**: 通过 Tanh + scale 限制残差幅度（默认 ±0.2），防止过度修改
3. **减轻过拟合**: 模型只学习小的微调量，而非从头学习整个映射

---

## 🚀 使用方法

### 1. 基础用法：残差学习模式（默认启用）

```bash
python baseline.py \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5
```

**默认配置**:
- ✅ 残差学习模式已启用 (`use_residual_learning=True`)
- 🔥 CLIP 完全可训练（端到端微调）
- 📏 残差缩放因子: `scale_q=0.2`, `scale_c=0.2`

---

### 2. 残差学习 + 部分冻结 CLIP（推荐）

冻结 CLIP 的前 N 层，只训练最后几层，进一步保护预训练知识：

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 18 \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5
```

**效果**:
- 🧊 前 18 层冻结（ViT-L-14 共 24 层）
- 🔥 最后 6 层可训练
- ✅ 残差学习自动启用
- 💾 模型保存为: `baseline_residual_partial_freeze_18L_best.pt`

**层数选择建议**:
- ViT-B/32 (12 层): `--freeze_layers 8` (训练最后 4 层)
- ViT-L/14 (24 层): `--freeze_layers 18-20` (训练最后 4-6 层)

---

### 3. 调整残差缩放因子

如果觉得模型调整幅度太小或太大，可以调整缩放因子：

```bash
python baseline.py \
    --residual_scale_q 0.3 \
    --residual_scale_c 0.3 \
    --epochs 20
```

**缩放因子说明**:
- `0.1`: 非常保守，几乎不偏离 CLIP（适合数据少的情况）
- `0.2`: **默认值**，平衡性能和稳定性
- `0.3-0.5`: 较激进，允许更大调整（适合数据充足的情况）
- `>0.5`: 不推荐，失去残差学习的意义

---

### 4. 禁用残差学习（回退到传统模式）

如果想使用传统的直接预测模式（不推荐）：

```bash
python baseline.py \
    --no_residual_learning \
    --epochs 20
```

---

### 5. 完全冻结 CLIP（Linear Probing）

只训练预测头，CLIP 完全冻结：

```bash
python baseline.py \
    --freeze_clip \
    --epochs 20 \
    --lr 1e-3  # 冻结时可以用更大的学习率
```

**注意**: 此模式下残差学习仍然有效，因为预测头本身使用残差架构。

---

## 📊 配置对比

| 模式 | 命令 | CLIP 状态 | 预测方式 | 适用场景 |
|------|------|-----------|----------|----------|
| **残差学习 + 部分冻结** | `--partial_freeze --freeze_layers 18` | 前 18 层冻结 | `cos(img,txt) + Δ` | 🏆 **推荐**，平衡性能和稳定性 |
| **残差学习 + 完全微调** | 默认 | 完全可训练 | `cos(img,txt) + Δ` | 数据充足，追求极致性能 |
| **Linear Probing** | `--freeze_clip` | 完全冻结 | `cos(img,txt) + Δ` | 快速验证，数据很少 |
| **传统模式** | `--no_residual_learning` | 完全可训练 | 直接预测 | 消融实验/对比基线 |

---

## 🔬 训练输出示例

启用残差学习后，训练时会显示详细配置：

```
======================================================================
Training Configuration:
  - Epochs: 20, Batch Size: 32, LR: 3e-05
  - Loss Weights: w_q=0.5, w_c=0.5
  - Use Explanations: False

  🔧 CLIP 冻结策略:
     🧊 部分冻结：前 18 层冻结，其余可训练

  🎯 预测架构:
     ✅ 残差学习模式（保留 CLIP 对齐空间）
        - Quality:  q = q_base + Δq × 0.2
        - Consistency: c = cos(img,txt) + Δc × 0.2
        - 原理: 只学习微调量，防止破坏 CLIP 原始空间
======================================================================
```

---

## 🎓 理论依据

### 为什么残差学习有效？

1. **预训练知识保护**: CLIP 在大规模数据上训练，其对齐空间已经很好，直接预测可能破坏这个空间
2. **梯度流动**: 残差连接提供直接的梯度通路，加速收敛
3. **归纳偏置**: 强制模型从 CLIP 基准出发，提供良好的初始点
4. **过拟合抑制**: 限制模型的表达能力（通过 scale），防止在小数据集上过拟合

### 数学形式

传统方式:
```
score = f(CLIP_features)  # 完全重新学习映射
```

残差学习:
```
score = CLIP_base_score + bounded_correction
      = g(CLIP_features) + tanh(h(CLIP_features)) × α
```
其中 α 是小的缩放因子（如 0.2），保证 `|correction| ≤ α`

---

## 💡 最佳实践

### 推荐配置 1: 数据充足（>3K 样本）

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 18 \
    --residual_scale_q 0.2 \
    --residual_scale_c 0.2 \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5
```

### 推荐配置 2: 数据较少（<1K 样本）

```bash
python baseline.py \
    --partial_freeze \
    --freeze_layers 20 \
    --residual_scale_q 0.1 \
    --residual_scale_c 0.1 \
    --epochs 30 \
    --batch_size 16 \
    --lr 1e-5
```

### 推荐配置 3: 快速验证

```bash
python baseline.py \
    --freeze_clip \
    --epochs 10 \
    --batch_size 64 \
    --lr 1e-3
```

---

## 📈 性能监控

训练过程中，注意观察：

1. **Base vs Final Score**: 
   - 如果 `Δ` 经常接近 `±scale`，说明缩放因子可能太小
   - 如果 `Δ` 都很小（<0.05），可能过度约束，可以增大 `scale`

2. **SROCC 曲线**:
   - 残差学习通常收敛更快（5-10 epoch）
   - 验证集性能更稳定（波动小）

3. **过拟合检测**:
   - Train-Val gap 应该 <0.05
   - 如果 gap 过大，考虑增大冻结层数或减小 `scale`

---

## ⚠️ 注意事项

1. **不要同时启用 `--use_refinement` 和残差学习**: 
   - 残差学习模式已经内置了更优雅的残差机制
   - 旧的 refinement module 会被自动禁用

2. **学习率建议**:
   - 完全微调: `1e-5 ~ 5e-5`
   - 部分冻结: `3e-5 ~ 1e-4`
   - 完全冻结: `1e-3 ~ 5e-3`

3. **Batch Size**:
   - 残差学习对 batch size 不敏感
   - 建议根据显存调整: 16/32/64

---

## 🔍 消融实验建议

对比不同配置的效果：

```bash
# 实验 1: 基准（传统模式）
python baseline.py --no_residual_learning --epochs 20

# 实验 2: 残差学习
python baseline.py --epochs 20

# 实验 3: 残差学习 + 部分冻结
python baseline.py --partial_freeze --freeze_layers 18 --epochs 20

# 实验 4: 完全冻结
python baseline.py --freeze_clip --epochs 20
```

预期结果:
- 实验 3 通常效果最好（平衡性能和泛化）
- 实验 2 在大数据集上可能略优
- 实验 4 最快，适合快速验证

---

## 📚 相关论文

1. **Residual Learning**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
2. **CLIP**: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision", ICML 2021
3. **Adapter Tuning**: Houlsby et al., "Parameter-Efficient Transfer Learning for NLP", ICML 2019

---

## 🆘 常见问题

**Q: 残差学习会降低性能吗？**  
A: 不会。在大多数情况下，残差学习提供更好的泛化和稳定性，SROCC 通常提升 1-3%。

**Q: 为什么不直接用 CLIP 的 cosine similarity？**  
A: CLIP 的 cosine similarity 对于 AIGC 图像不够精细，需要学习一个小的修正量来适应特定任务。

**Q: 可以动态调整 scale 吗？**  
A: 目前 scale 是固定的。未来可以考虑将其作为可学习参数，或使用课程学习逐步增大。

**Q: 部分冻结时哪些层应该冻结？**  
A: 浅层学习通用特征（边缘、纹理），深层学习任务相关特征。通常冻结前 70-80% 的层效果最好。

---

## 📝 代码结构

新增的关键模块：

```python
# baseline.py

class BaselineCLIPScore:
    def __init__(..., use_residual_learning=True, residual_scale_q=0.2, ...):
        if use_residual_learning:
            # Quality 分支
            self.q_base_head = nn.Linear(dim, 1)  # 基准分数
            self.q_delta_head = nn.Sequential(...)  # 残差预测器
            
            # Consistency 分支
            self.c_base_scale = nn.Parameter(...)  # CLIP cos 的可学习缩放
            self.c_delta_head = nn.Sequential(...)  # 残差预测器
    
    def forward(self, ...):
        if self.use_residual_learning:
            # Quality: base + delta
            q = clamp(q_base + q_delta * scale_q, 0, 1)
            
            # Consistency: cos + delta
            c = clamp(cos_sim + c_delta * scale_c, 0, 1)
        ...
    
    def _partial_freeze_clip(self, freeze_layers):
        # 冻结前 N 层编码器
        ...
```

---

**最后更新**: 2025-10-16  
**版本**: 1.0

