#!/bin/bash
# 残差学习模式训练脚本
# 用法: bash run_residual_training.sh

echo "=================================="
echo "残差学习 + 部分冻结 CLIP 训练"
echo "=================================="
echo ""
echo "📝 训练配置："
echo "  - 模式: 残差学习（保留 CLIP 对齐空间）"
echo "  - 冻结: 前 18 层（ViT-L-14 共 24 层）"
echo "  - 残差缩放: q=0.2, c=0.2"
echo "  - Epochs: 20"
echo "  - Batch Size: 32"
echo "  - Learning Rate: 3e-5"
echo ""
echo "💡 核心原理："
echo "  Quality:  q = q_base + Δq × 0.2"
echo "  Consistency: c = cos(img,txt) + Δc × 0.2"
echo ""
echo "开始训练..."
echo ""

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

echo ""
echo "=================================="
echo "训练完成！"
echo "模型保存为: baseline_residual_partial_freeze_18L_best.pt"
echo "=================================="

