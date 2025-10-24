# 🚀 快速开始 - 新版本使用指南

> 本指南介绍如何使用项目的新功能（v1.1.0）

---

## 📋 目录

- [安装依赖](#安装依赖)
- [完整工作流程](#完整工作流程)
- [快速示例](#快速示例)
- [常见任务](#常见任务)

---

## 📦 安装依赖

```bash
# 克隆仓库
git clone https://github.com/qiyan111/AIGC.git
cd AIGC

# 安装依赖
pip install -r requirements.txt
```

---

## 🎯 完整工作流程

### 步骤 1: 数据分析（可选但推荐）

```bash
# 分析数据集，了解数据分布
python data_utils.py analyze \
    --csv data/data.csv \
    --image_dir data/ACGIQA-3K \
    --output_dir data_analysis
```

**输出**：
- 数据集统计信息
- 分数分布图
- 相关性分析
- 数据质量报告

---

### 步骤 2: 训练模型

#### 方式 A: 使用命令行参数

```bash
python train.py \
    --data_csv_path data/data.csv \
    --image_base_dir data/ACGIQA-3K \
    --partial_freeze \
    --freeze_layers 18 \
    --epochs 20 \
    --batch_size 32 \
    --mixed_precision \
    --experiment_name "my_first_experiment"
```

#### 方式 B: 使用配置文件（推荐）

```bash
# 使用预设配置
python train.py --config configs/baseline_residual.json

# 或者先创建自定义配置
python train.py \
    --partial_freeze \
    --freeze_layers 18 \
    --mixed_precision \
    --save_config configs/my_config.json

# 然后使用配置训练
python train.py --config configs/my_config.json
```

**优势**：
- ⚡ 混合精度训练加速 1.5-2x
- 📝 完整的训练日志
- 💾 自动保存最佳模型
- 📊 实时训练监控

---

### 步骤 3: 推理预测

#### 单张图像测试

```bash
python inference.py \
    --model_path checkpoints/baseline_residual_best.pt \
    --single test_image.jpg "a beautiful landscape with mountains"
```

**输出示例**：
```
==================================================================================
📊 预测结果:
----------------------------------------------------------------------------------
  🎨 图像质量 (Quality):         4.35 / 5.00
  🔗 文本一致性 (Consistency):   4.52 / 5.00
  📈 基准分数 (Coarse/Base):     4.40 / 5.00
  ➕ 残差修正 (Residual):        +0.12
==================================================================================
```

#### 批量推理

```bash
python inference.py \
    --model_path checkpoints/baseline_residual_best.pt \
    --csv data/data.csv data/ACGIQA-3K \
    --output predictions.csv \
    --batch_size 64
```

---

### 步骤 4: 评估模型

```bash
python evaluate.py \
    --predictions_csv predictions.csv \
    --output_dir evaluation_results
```

**生成的文件**：
- `scatter_plot.png` - 预测 vs 真实值散点图
- `error_distribution.png` - 误差分布分析
- `score_distribution.png` - 分数分布对比
- `evaluation_report.txt` - 详细评估报告

**评估报告示例**：
```
🎯 Quality 评估指标:
  • SROCC (Spearman):    0.9012
  • PLCC (Pearson):      0.8956
  • MAE:                 0.2834
  • RMSE:                0.3621

🔗 Consistency 评估指标:
  • SROCC (Spearman):    0.8789
  • PLCC (Pearson):      0.8701
  • MAE:                 0.3102
  • RMSE:                0.3945

💡 建议:
  ✅ 模型性能优秀！可以考虑在更大数据集上测试
```

---

## ⚡ 快速示例

### 示例 1: 10 分钟快速训练和评估

```bash
# 1. 快速训练（使用快速配置）
python train.py --config configs/fast_training.json --epochs 10

# 2. 批量推理
python inference.py \
    --model_path checkpoints/fast_training_fp16_best.pt \
    --csv data/data.csv data/ACGIQA-3K \
    --output quick_results.csv

# 3. 评估
python evaluate.py --predictions_csv quick_results.csv
```

---

### 示例 2: 小数据集训练

```bash
# 使用小数据集配置（完全冻结CLIP）
python train.py --config configs/small_dataset.json
```

---

### 示例 3: 数据预处理 + 训练 + 评估

```bash
# 1. 数据预处理（移除异常值）
python data_utils.py preprocess \
    --csv data/data.csv \
    --output data/data_cleaned.csv \
    --remove_outliers

# 2. 训练（使用清理后的数据）
python train.py \
    --data_csv_path data/data_cleaned.csv \
    --image_base_dir data/ACGIQA-3K \
    --config configs/baseline_residual.json

# 3. 评估
python inference.py \
    --model_path checkpoints/baseline_residual_best.pt \
    --csv data/data_cleaned.csv data/ACGIQA-3K \
    --output results_cleaned.csv

python evaluate.py --predictions_csv results_cleaned.csv
```

---

## 📚 常见任务

### 任务 1: 调整超参数

```bash
# 创建自定义配置
python train.py \
    --partial_freeze \
    --freeze_layers 20 \
    --residual_scale_q 0.25 \
    --residual_scale_c 0.25 \
    --lr 5e-5 \
    --batch_size 64 \
    --save_config configs/custom_config.json

# 使用自定义配置训练
python train.py --config configs/custom_config.json --mixed_precision
```

---

### 任务 2: 对比不同配置

```bash
# 实验 1: 基线配置
python train.py \
    --config configs/baseline_residual.json \
    --experiment_name "exp1_baseline"

# 实验 2: 更大的残差缩放
python train.py \
    --config configs/baseline_residual.json \
    --residual_scale_q 0.3 \
    --residual_scale_c 0.3 \
    --experiment_name "exp2_larger_residual"

# 实验 3: 更多可训练层
python train.py \
    --config configs/baseline_residual.json \
    --freeze_layers 16 \
    --experiment_name "exp3_more_trainable"

# 对比评估
for exp in exp1_baseline exp2_larger_residual exp3_more_trainable; do
    python inference.py \
        --model_path checkpoints/${exp}_best.pt \
        --csv data/data.csv data/ACGIQA-3K \
        --output ${exp}_results.csv
    
    python evaluate.py \
        --predictions_csv ${exp}_results.csv \
        --output_dir evaluation_${exp}
done
```

---

### 任务 3: 分析数据集

```bash
# 完整数据分析
python data_utils.py analyze \
    --csv data/data.csv \
    --image_dir data/ACGIQA-3K \
    --output_dir data_analysis

# 查看生成的图表
ls data_analysis/
# score_distributions.png
# score_correlation.png
```

---

### 任务 4: 生产环境推理

```bash
# 创建推理脚本
cat > batch_inference.sh << 'EOF'
#!/bin/bash
MODEL="checkpoints/baseline_residual_best.pt"
INPUT_CSV="$1"
IMAGE_DIR="$2"
OUTPUT_CSV="$3"

python inference.py \
    --model_path "$MODEL" \
    --csv "$INPUT_CSV" "$IMAGE_DIR" \
    --output "$OUTPUT_CSV" \
    --batch_size 128

echo "✅ 推理完成: $OUTPUT_CSV"
EOF

chmod +x batch_inference.sh

# 使用
./batch_inference.sh data/test.csv data/test_images predictions.csv
```

---

## 🎛️ 配置对比

### 不同场景推荐配置

| 场景 | 配置文件 | 关键参数 | 训练时间 |
|------|----------|----------|----------|
| 小数据集 (< 1K) | `small_dataset.json` | freeze_clip=True | ~10 min |
| 标准训练 (1-5K) | `baseline_residual.json` | partial_freeze=True | ~20 min |
| 快速实验 | `fast_training.json` | mixed_precision=True | ~15 min |

---

## 🔧 调试技巧

### 查看训练日志

```bash
# 实时查看日志
tail -f logs/baseline_residual_*.log

# 搜索错误
grep "ERROR" logs/baseline_residual_*.log
```

### 显存不足

```bash
# 减小 batch size
python train.py --config configs/baseline_residual.json --batch_size 16

# 或使用混合精度训练（减少显存占用约40%）
python train.py --config configs/baseline_residual.json --mixed_precision
```

### 训练不稳定

```bash
# 增大 warmup
python train.py \
    --config configs/baseline_residual.json \
    --warmup_ratio 0.1 \
    --max_grad_norm 0.5
```

---

## 📊 性能对比

### 原版 vs 新版

| 特性 | 原版 (baseline.py) | 新版 (train.py) |
|------|-------------------|-----------------|
| 训练速度 | 1x | 1.5-2x (混合精度) |
| 配置管理 | ❌ | ✅ |
| 日志系统 | 基础 | 完善 |
| 错误提示 | 简单 | 详细 |
| 进度监控 | ❌ | ✅ |
| 推理工具 | ❌ | ✅ |
| 评估工具 | 基础 | 完善 |

---

## 💡 最佳实践

### 1. 开始新项目

```bash
# 1. 先分析数据
python data_utils.py analyze --csv data/data.csv --image_dir data/images

# 2. 根据数据量选择配置
# < 1K: configs/small_dataset.json
# 1-5K: configs/baseline_residual.json
# > 5K: 调整 residual_scale

# 3. 快速验证
python train.py --config configs/baseline_residual.json --epochs 5

# 4. 完整训练
python train.py --config configs/baseline_residual.json --mixed_precision

# 5. 评估
python inference.py --model_path checkpoints/xxx_best.pt --csv data/data.csv data/images --output results.csv
python evaluate.py --predictions_csv results.csv
```

### 2. 超参数调优

```bash
# 创建基线
python train.py --config configs/baseline_residual.json --experiment_name baseline

# 调整学习率
for lr in 1e-5 3e-5 5e-5 1e-4; do
    python train.py \
        --config configs/baseline_residual.json \
        --lr $lr \
        --experiment_name "lr_${lr}"
done

# 对比结果
ls logs/
ls checkpoints/
```

### 3. 生产部署

```bash
# 1. 训练最终模型
python train.py --config configs/baseline_residual.json --epochs 30 --experiment_name production

# 2. 完整评估
python inference.py \
    --model_path checkpoints/production_best.pt \
    --csv data/test.csv data/test_images \
    --output production_results.csv

python evaluate.py --predictions_csv production_results.csv

# 3. 保存配置和模型
cp configs/baseline_residual.json checkpoints/production_config.json
cp checkpoints/production_best.pt checkpoints/production_final.pt

# 4. 创建推理服务（TODO: 在 v1.2.0 实现）
```

---

## ❓ 常见问题

### Q: 如何选择配置文件？

**A**: 根据数据量：
- < 1K 样本: `configs/small_dataset.json`
- 1-5K 样本: `configs/baseline_residual.json` ⭐ 推荐
- > 5K 样本: 修改 `baseline_residual.json`，增大 `residual_scale`

### Q: 混合精度训练有什么要求？

**A**: 需要：
- NVIDIA GPU with Compute Capability >= 7.0 (如 RTX 20/30/40 系列, V100, A100)
- PyTorch >= 1.6

### Q: 如何使用多 GPU 训练？

**A**: 当前版本暂不支持，计划在 v1.2.0 添加。可以临时使用：
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch ...
```

### Q: 原来的 baseline.py 还能用吗？

**A**: 可以！所有原有功能完全保留，新功能是额外添加的。

---

## 📖 更多文档

- [完整 README](README.md)
- [改进总结](IMPROVEMENTS.md)
- [参数参考](PARAMETERS_QUICK_REFERENCE.md)
- [超参数调优](HYPERPARAMETER_TUNING_GUIDE.md)
- [残差学习原理](RESIDUAL_LEARNING_USAGE.md)

---

## 🎉 开始使用

```bash
# 最简单的开始方式
python train.py --config configs/fast_training.json --epochs 5
```

**祝训练顺利！** 🚀

如有问题，欢迎提 [Issue](https://github.com/qiyan111/AIGC/issues)。
