#!/bin/bash
# 残差学习消融实验
# 对比不同配置的效果

echo "========================================================================"
echo "残差学习消融实验 - 对比 4 种配置"
echo "========================================================================"
echo ""

# ===== 实验 1: 基准（传统模式，不使用残差学习）=====
echo "🔵 实验 1/4: 传统模式（基准）"
echo "   - 不使用残差学习"
echo "   - CLIP 完全可训练"
echo "   - 模型: baseline_best.pt"
echo ""

python baseline.py \
    --data_csv_path data/data.csv \
    --image_base_dir data/ACGIQA-3K \
    --output_dir outputs \
    --no_residual_learning \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5 \
    --w_q 0.5 \
    --w_c 0.5

echo ""
echo "✅ 实验 1 完成"
echo "------------------------------------------------------------------------"
echo ""

# ===== 实验 2: 残差学习（完全微调）=====
echo "🟢 实验 2/4: 残差学习 + 完全微调"
echo "   - ✅ 使用残差学习"
echo "   - CLIP 完全可训练"
echo "   - 模型: baseline_residual_best.pt"
echo ""

python baseline.py \
    --data_csv_path data/data.csv \
    --image_base_dir data/ACGIQA-3K \
    --output_dir outputs \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5 \
    --w_q 0.5 \
    --w_c 0.5

echo ""
echo "✅ 实验 2 完成"
echo "------------------------------------------------------------------------"
echo ""

# ===== 实验 3: 残差学习 + 部分冻结（推荐）=====
echo "🟡 实验 3/4: 残差学习 + 部分冻结 ⭐ 推荐"
echo "   - ✅ 使用残差学习"
echo "   - 🧊 前 18 层冻结"
echo "   - 模型: baseline_residual_partial_freeze_18L_best.pt"
echo ""

python baseline.py \
    --data_csv_path data/data.csv \
    --image_base_dir data/ACGIQA-3K \
    --output_dir outputs \
    --partial_freeze \
    --freeze_layers 18 \
    --epochs 20 \
    --batch_size 32 \
    --lr 3e-5 \
    --w_q 0.5 \
    --w_c 0.5

echo ""
echo "✅ 实验 3 完成"
echo "------------------------------------------------------------------------"
echo ""

# ===== 实验 4: 完全冻结 CLIP（Linear Probing）=====
echo "🔴 实验 4/4: Linear Probing（完全冻结 CLIP）"
echo "   - ✅ 使用残差学习"
echo "   - ❄️  CLIP 完全冻结"
echo "   - 更大学习率: 1e-3"
echo "   - 模型: baseline_residual_best.pt"
echo ""

python baseline.py \
    --data_csv_path data/data.csv \
    --image_base_dir data/ACGIQA-3K \
    --output_dir outputs \
    --freeze_clip \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --w_q 0.5 \
    --w_c 0.5

echo ""
echo "✅ 实验 4 完成"
echo "------------------------------------------------------------------------"
echo ""

echo "========================================================================"
echo "消融实验全部完成！"
echo "========================================================================"
echo ""
echo "📊 生成的模型文件："
echo "   1. baseline_best.pt                               (传统模式)"
echo "   2. baseline_residual_best.pt                      (残差学习)"
echo "   3. baseline_residual_partial_freeze_18L_best.pt   (残差 + 部分冻结) ⭐"
echo "   4. baseline_residual_best.pt                      (完全冻结)"
echo ""
echo "💡 下一步："
echo "   1. 对比验证集 SROCC_Q 和 SROCC_C"
echo "   2. 分析训练曲线（loss 和 SROCC）"
echo "   3. 检查过拟合情况（train-val gap）"
echo "   4. 选择最佳配置用于最终训练"
echo ""
echo "预期结果:"
echo "   - 实验 3（残差 + 部分冻结）通常效果最好"
echo "   - 实验 2 在大数据集上可能略优于实验 3"
echo "   - 实验 1 可能过拟合（train-val gap 大）"
echo "   - 实验 4 最快，但性能可能略低"
echo "========================================================================"

