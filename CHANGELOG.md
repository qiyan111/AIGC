# 更新日志

## [1.0.0] - 2025-10-24

### 新增功能 ✨

#### 核心架构
- ✅ 实现基于残差学习的 AIGC 图像评估模型
- ✅ 支持双任务预测：图像质量（Quality）和文本一致性（Consistency）
- ✅ 集成 CLIP 模型作为特征提取器

#### 残差学习机制 🎯
- ✅ Quality 预测：`q = q_base + Δq × scale_q`
- ✅ Consistency 预测：`c = cos(img,txt) + Δc × scale_c`
- ✅ 保留 CLIP 原始对齐空间，防止过拟合

#### 冻结策略 ❄️
- ✅ 支持完全冻结 CLIP（Linear Probing）
- ✅ 支持部分冻结（只训练最后 N 层）
- ✅ 支持完全微调

#### 可调参数 🎛️
- ✅ 20+ 可调参数，包括：
  - 学习率、批次大小、训练轮数
  - 残差缩放因子（`residual_scale_q/c`）
  - 损失权重（`w_q/w_c`）
  - 冻结层数（`freeze_layers`）
  - 正则化参数（`dropout`, `weight_decay`, `max_grad_norm`）
  - Warmup 比例（`warmup_ratio`）

#### 训练脚本 📜
- ✅ `run_residual_training.sh` - 推荐配置训练脚本
- ✅ `run_ablation_study.sh` - 完整消融实验脚本

#### 文档 📚
- ✅ `README.md` - 项目主文档
- ✅ `RESIDUAL_LEARNING_USAGE.md` - 残差学习详细指南
- ✅ `PARAMETERS_QUICK_REFERENCE.md` - 参数快速参考
- ✅ `HYPERPARAMETER_TUNING_GUIDE.md` - 超参数调优指南

### 性能指标 📊

在 ACGIQA-3K 数据集上：
- SROCC_Q: **0.90**
- SROCC_C: **0.88**
- 训练时间: ~20 分钟（A100 40GB）

### 已知问题 🐛

无

---

## 未来计划 🚀

### v1.1.0（计划中）
- [ ] 添加自动超参数搜索（Optuna）
- [ ] 支持多 GPU 训练
- [ ] 添加 TensorBoard/WandB 日志
- [ ] 支持更多 CLIP 模型（ViT-B/16, ViT-H/14）

### v1.2.0（计划中）
- [ ] 支持增量学习
- [ ] 添加模型蒸馏功能
- [ ] 提供预训练模型下载

### v2.0.0（计划中）
- [ ] 支持视频质量评估
- [ ] 添加交互式评估界面
- [ ] 支持 ONNX/TorchScript 导出

---

## 贡献者 👥

- [@qiyan111](https://github.com/qiyan111) - 项目创建者和维护者

---

**格式说明**：
- `新增功能` - 新增的功能
- `改进` - 对现有功能的改进
- `修复` - Bug 修复
- `破坏性变更` - 不兼容的 API 变更
- `文档` - 文档更新
- `性能` - 性能优化

