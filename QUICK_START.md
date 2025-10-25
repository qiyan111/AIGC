# 快速开始指南

## 5 分钟快速上手

### 步骤 1: 安装依赖

```bash
pip install -r requirements.txt
```

### 步骤 2: 准备数据

创建 CSV 文件（例如 `data.csv`）：

```csv
name,prompt,mos_quality,mos_align
image1.jpg,a beautiful sunset,4.5,4.2
image2.jpg,a cat sitting on a chair,3.8,4.0
...
```

**必需字段**：
- `name`: 图像文件名
- `prompt`: 文本提示词
- `mos_quality`: 图像质量分数（1-5）
- `mos_align`: 文本一致性分数（1-5）

### 步骤 3: 下载 CLIP 模型

```bash
# 方法 1: 自动下载（需要联网）
# 代码会自动从 Hugging Face 下载

# 方法 2: 手动下载
# 从 https://huggingface.co/openai/clip-vit-large-patch14 下载模型
# 放置到本地目录，例如 ./clip-vit-large-patch14/
```

### 步骤 4: 运行训练

```bash
python baseline.py \
    --data_csv_path data/data.csv \
    --image_base_dir data/ACGIQA-3K \
    --output_dir outputs \
    --partial_freeze \
    --freeze_layers 18 \
    --epochs 20
```

**或者使用推荐脚本**：

```bash
# 编辑 run_residual_training.sh，修改数据路径
bash run_residual_training.sh
```

---

## 常见使用场景

### 场景 1: 我有很少的数据（< 500 张图）

```bash
python baseline.py \
    --data_csv_path data.csv \
    --image_base_dir images/ \
    --freeze_clip \
    --epochs 30 \
    --lr 1e-3 \
    --residual_scale_q 0.1 \
    --residual_scale_c 0.1
```

**说明**：完全冻结 CLIP，只训练预测头，防止过拟合。

---

### 场景 2: 我有中等数据（1K-5K 张图）⭐ 推荐

```bash
python baseline.py \
    --data_csv_path data.csv \
    --image_base_dir images/ \
    --partial_freeze \
    --freeze_layers 18 \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5
```

**说明**：部分冻结 CLIP，平衡性能和泛化能力。

---

### 场景 3: 我有大量数据（> 5K 张图）

```bash
python baseline.py \
    --data_csv_path data.csv \
    --image_base_dir images/ \
    --residual_scale_q 0.3 \
    --residual_scale_c 0.3 \
    --epochs 15 \
    --batch_size 64 \
    --lr 3e-5
```

**说明**：完全微调 CLIP，允许更大的调整幅度。

---

### 场景 4: 我更关注图像质量

```bash
python baseline.py \
    --data_csv_path data.csv \
    --image_base_dir images/ \
    --w_q 0.7 \
    --w_c 0.3 \
    --residual_scale_q 0.3 \
    --residual_scale_c 0.15
```

**说明**：增大 Quality 损失权重和残差缩放。

---

### 场景 5: 我更关注文本一致性

```bash
python baseline.py \
    --data_csv_path data.csv \
    --image_base_dir images/ \
    --w_q 0.3 \
    --w_c 0.7 \
    --residual_scale_q 0.15 \
    --residual_scale_c 0.3
```

**说明**：增大 Consistency 损失权重和残差缩放。

---

## 训练输出示例

```
======================================================================
Training Configuration:
  - Epochs: 20, Batch Size: 32, LR: 3e-05
  - Optimizer: AdamW (weight_decay=0.0001, warmup_ratio=0.05)
  - Regularization: dropout=0.1, max_grad_norm=1.0
  - Loss Weights: w_q=0.5, w_c=0.5

  🔧 CLIP 冻结策略:
     🧊 部分冻结：前 18 层冻结，其余可训练

  🎯 预测架构:
     ✅ 残差学习模式（保留 CLIP 对齐空间）
        - Quality:  q = q_base + Δq × 0.2
        - Consistency: c = cos(img,txt) + Δc × 0.2
        - 原理: 只学习微调量，防止破坏 CLIP 原始空间
======================================================================

Ep1 TrainLoss=0.0234(Q0.0118,C0.0116)  Val SROCC_Q=0.8523,SROCC_C=0.8312
[Checkpoint] New best SROCC_C=0.8312 -> baseline_residual_partial_freeze_18L_best.pt
Ep2 TrainLoss=0.0198(Q0.0099,C0.0099)  Val SROCC_Q=0.8678,SROCC_C=0.8501
[Checkpoint] New best SROCC_C=0.8501 -> baseline_residual_partial_freeze_18L_best.pt
...
Ep20 TrainLoss=0.0087(Q0.0043,C0.0044)  Val SROCC_Q=0.9012,SROCC_C=0.8789
[Checkpoint] New best SROCC_C=0.8789 -> baseline_residual_partial_freeze_18L_best.pt

Training Complete!
Best SROCC_C: 0.8789
```

---

## 模型保存

训练完成后，最佳模型会保存为：

- **残差学习 + 部分冻结**: `baseline_residual_partial_freeze_18L_best.pt`
- **残差学习**: `baseline_residual_best.pt`
- **传统模式**: `baseline_best.pt`

---

## 加载和使用模型

```python
import torch
from baseline import BaselineCLIPScore

# 加载模型
model = BaselineCLIPScore(
    clip_model_name="openai/clip-vit-large-patch14",
    use_residual_learning=True,
    partial_freeze=True,
    freeze_layers=18
)

# 加载权重
model.load_state_dict(torch.load("baseline_residual_partial_freeze_18L_best.pt"))
model.eval()

# 推理（需要准备 pixel_values, input_ids, attention_mask）
with torch.no_grad():
    q_pred, c_pred, _, _, _ = model(pixel_values, input_ids, attention_mask)
    
print(f"Quality Score: {q_pred.item() * 5:.2f}/5")
print(f"Consistency Score: {c_pred.item() * 5:.2f}/5")
```

---

## 常见问题

### Q: 训练过程中显存不足怎么办？

**A**: 减小 batch size：
```bash
--batch_size 16  # 或 8
```

### Q: 训练速度太慢？

**A**: 
1. 使用完全冻结：`--freeze_clip`
2. 减少 epochs：`--epochs 10`
3. 增大 batch size（如果显存允许）：`--batch_size 64`

### Q: 验证集性能不提升？

**A**: 可能过拟合，尝试：
```bash
--freeze_layers 20  # 增加冻结层数
--dropout 0.2       # 增大 dropout
--weight_decay 5e-4 # 增大正则化
```

### Q: 如何查看所有可用参数？

**A**: 
```bash
python baseline.py --help
```

或查看文档：`PARAMETERS_QUICK_REFERENCE.md`

---

## 下一步

- 📖 阅读 [残差学习原理](RESIDUAL_LEARNING_USAGE.md)
- 🎛️ 查看 [完整参数列表](PARAMETERS_QUICK_REFERENCE.md)
- 🔧 学习 [超参数调优](HYPERPARAMETER_TUNING_GUIDE.md)
- 🧪 运行 [消融实验](run_ablation_study.sh)

---

**祝训练顺利！如有问题，欢迎提 Issue。** 🎉

