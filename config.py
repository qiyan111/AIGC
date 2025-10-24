#!/usr/bin/env python3
"""
配置文件管理系统
用于统一管理所有超参数和路径配置，避免硬编码
"""

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class DataConfig:
    """数据相关配置"""
    # 数据路径
    data_csv_path: str = "data/data.csv"
    image_base_dir: str = "data/ACGIQA-3K"
    clip_model_name: str = "openai/clip-vit-large-patch14"
    
    # 数据划分
    test_size: float = 0.2
    random_seed: int = 42
    
    # 图像预处理
    image_size: int = 224
    
    def validate(self):
        """验证配置是否有效"""
        if not os.path.exists(self.data_csv_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_csv_path}")
        if not os.path.exists(self.image_base_dir):
            raise FileNotFoundError(f"图像目录不存在: {self.image_base_dir}")


@dataclass
class ModelConfig:
    """模型相关配置"""
    # CLIP 冻结策略
    freeze_clip: bool = False  # 完全冻结 CLIP（Linear Probing）
    partial_freeze: bool = False  # 部分冻结（只训练最后几层）
    freeze_layers: int = 8  # 冻结前 N 层（ViT-L-14 有 24 层）
    
    # 残差学习
    use_residual_learning: bool = True  # 启用残差学习
    residual_scale_q: float = 0.2  # Quality 残差缩放因子
    residual_scale_c: float = 0.2  # Consistency 残差缩放因子
    
    # 网络结构
    dropout: float = 0.1  # Dropout 比例
    use_two_branch: bool = False  # 使用两分支结构
    
    # 可选功能
    use_refinement: bool = False  # 启用精炼模块
    refinement_layers: int = 4
    refinement_heads: int = 8
    refinement_dim: int = 256
    strict_residual: bool = True
    
    use_explanations: bool = False  # 启用解释性学习
    explanation_column: str = "explanation"


@dataclass
class TrainingConfig:
    """训练相关配置"""
    # 训练超参数
    epochs: int = 20
    batch_size: int = 32
    lr: float = 3e-5
    weight_decay: float = 1e-4
    
    # 学习率调度
    warmup_ratio: float = 0.05  # Warmup 步数占比
    
    # 正则化
    max_grad_norm: float = 1.0  # 梯度裁剪阈值，0 表示不裁剪
    
    # 损失权重
    w_q: float = 0.5  # Quality 损失权重
    w_c: float = 0.5  # Consistency 损失权重
    w_exp: float = 0.1  # Explanation 损失权重
    
    # 训练设置
    device: str = "cuda"  # cuda 或 cpu
    mixed_precision: bool = False  # 混合精度训练（自动加速）
    num_workers: int = 4  # DataLoader 工作线程数
    
    # 检查点
    save_dir: str = "checkpoints"
    save_best_only: bool = True
    
    # 日志
    log_dir: str = "logs"
    log_interval: int = 10  # 每 N 个 batch 输出一次日志
    
    def __post_init__(self):
        """初始化后处理"""
        # 创建必要的目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class Config:
    """总配置类，包含所有子配置"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 实验配置
    experiment_name: str = "baseline"
    notes: str = ""  # 实验备注
    
    def validate(self):
        """验证所有配置"""
        self.data.validate()
    
    def save(self, path: str):
        """保存配置到 JSON 文件"""
        config_dict = {
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "experiment_name": self.experiment_name,
            "notes": self.notes
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"✅ 配置已保存到: {path}")
    
    @classmethod
    def load(cls, path: str):
        """从 JSON 文件加载配置"""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = cls()
        # 更新数据配置
        for key, value in config_dict.get("data", {}).items():
            if hasattr(config.data, key):
                setattr(config.data, key, value)
        # 更新模型配置
        for key, value in config_dict.get("model", {}).items():
            if hasattr(config.model, key):
                setattr(config.model, key, value)
        # 更新训练配置
        for key, value in config_dict.get("training", {}).items():
            if hasattr(config.training, key):
                setattr(config.training, key, value)
        
        config.experiment_name = config_dict.get("experiment_name", "baseline")
        config.notes = config_dict.get("notes", "")
        
        print(f"✅ 配置已加载: {path}")
        return config
    
    def print_summary(self):
        """打印配置摘要"""
        print("\n" + "=" * 80)
        print(f"🔧 实验配置: {self.experiment_name}")
        if self.notes:
            print(f"📝 备注: {self.notes}")
        print("=" * 80)
        
        print("\n📊 数据配置:")
        print(f"  • CSV 文件: {self.data.data_csv_path}")
        print(f"  • 图像目录: {self.data.image_base_dir}")
        print(f"  • CLIP 模型: {self.data.clip_model_name}")
        print(f"  • 测试集比例: {self.data.test_size * 100:.0f}%")
        
        print("\n🎯 模型配置:")
        if self.model.freeze_clip:
            print(f"  • 冻结策略: ❄️  完全冻结 CLIP (Linear Probing)")
        elif self.model.partial_freeze:
            print(f"  • 冻结策略: 🧊 部分冻结 (前 {self.model.freeze_layers} 层)")
        else:
            print(f"  • 冻结策略: 🔥 完全可训练 (端到端微调)")
        
        if self.model.use_residual_learning:
            print(f"  • 预测架构: ✅ 残差学习模式")
            print(f"    - Quality:  q = q_base + Δq × {self.model.residual_scale_q}")
            print(f"    - Consistency: c = cos(img,txt) + Δc × {self.model.residual_scale_c}")
        else:
            print(f"  • 预测架构: ⚠️  传统模式")
        
        print(f"  • Dropout: {self.model.dropout}")
        
        print("\n🚀 训练配置:")
        print(f"  • Epochs: {self.training.epochs}")
        print(f"  • Batch Size: {self.training.batch_size}")
        print(f"  • Learning Rate: {self.training.lr}")
        print(f"  • Weight Decay: {self.training.weight_decay}")
        print(f"  • Warmup Ratio: {self.training.warmup_ratio}")
        print(f"  • Max Grad Norm: {self.training.max_grad_norm}")
        print(f"  • 损失权重: w_q={self.training.w_q}, w_c={self.training.w_c}")
        print(f"  • 设备: {self.training.device}")
        if self.training.mixed_precision:
            print(f"  • ⚡ 混合精度训练: 已启用 (加速 ~1.5-2x)")
        
        print("=" * 80 + "\n")


def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="AIGC 图像质量评估模型训练",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置文件
    parser.add_argument('--config', type=str, help='从 JSON 文件加载配置')
    parser.add_argument('--save_config', type=str, help='保存当前配置到 JSON 文件')
    
    # 实验配置
    parser.add_argument('--experiment_name', type=str, help='实验名称')
    parser.add_argument('--notes', type=str, help='实验备注')
    
    # 数据配置
    data_group = parser.add_argument_group('数据配置')
    data_group.add_argument('--data_csv_path', type=str, help='数据 CSV 文件路径')
    data_group.add_argument('--image_base_dir', type=str, help='图像目录路径')
    data_group.add_argument('--clip_model_name', type=str, help='CLIP 模型名称或路径')
    
    # 模型配置
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--freeze_clip', action='store_true', help='完全冻结 CLIP')
    model_group.add_argument('--partial_freeze', action='store_true', help='部分冻结 CLIP')
    model_group.add_argument('--freeze_layers', type=int, help='冻结前 N 层')
    model_group.add_argument('--no_residual_learning', action='store_true', help='禁用残差学习')
    model_group.add_argument('--residual_scale_q', type=float, help='Quality 残差缩放因子')
    model_group.add_argument('--residual_scale_c', type=float, help='Consistency 残差缩放因子')
    model_group.add_argument('--dropout', type=float, help='Dropout 比例')
    model_group.add_argument('--use_two_branch', action='store_true', help='使用两分支结构')
    
    # 训练配置
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--epochs', type=int, help='训练轮数')
    train_group.add_argument('--batch_size', type=int, help='批次大小')
    train_group.add_argument('--lr', type=float, help='学习率')
    train_group.add_argument('--weight_decay', type=float, help='权重衰减')
    train_group.add_argument('--warmup_ratio', type=float, help='Warmup 比例')
    train_group.add_argument('--max_grad_norm', type=float, help='梯度裁剪阈值')
    train_group.add_argument('--w_q', type=float, help='Quality 损失权重')
    train_group.add_argument('--w_c', type=float, help='Consistency 损失权重')
    train_group.add_argument('--mixed_precision', action='store_true', help='启用混合精度训练')
    train_group.add_argument('--save_dir', type=str, help='模型保存目录')
    
    # 可选功能
    optional_group = parser.add_argument_group('可选功能')
    optional_group.add_argument('--use_refinement', action='store_true', help='启用精炼模块')
    optional_group.add_argument('--use_explanations', action='store_true', help='启用解释性学习')
    
    return parser


def parse_args_to_config(args: argparse.Namespace) -> Config:
    """将命令行参数解析为配置对象"""
    # 如果指定了配置文件，先加载
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()
    
    # 覆盖配置（命令行参数优先）
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.notes:
        config.notes = args.notes
    
    # 数据配置
    if args.data_csv_path:
        config.data.data_csv_path = args.data_csv_path
    if args.image_base_dir:
        config.data.image_base_dir = args.image_base_dir
    if args.clip_model_name:
        config.data.clip_model_name = args.clip_model_name
    
    # 模型配置
    if args.freeze_clip:
        config.model.freeze_clip = True
    if args.partial_freeze:
        config.model.partial_freeze = True
    if args.freeze_layers is not None:
        config.model.freeze_layers = args.freeze_layers
    if args.no_residual_learning:
        config.model.use_residual_learning = False
    if args.residual_scale_q is not None:
        config.model.residual_scale_q = args.residual_scale_q
    if args.residual_scale_c is not None:
        config.model.residual_scale_c = args.residual_scale_c
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.use_two_branch:
        config.model.use_two_branch = True
    
    # 训练配置
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.weight_decay is not None:
        config.training.weight_decay = args.weight_decay
    if args.warmup_ratio is not None:
        config.training.warmup_ratio = args.warmup_ratio
    if args.max_grad_norm is not None:
        config.training.max_grad_norm = args.max_grad_norm
    if args.w_q is not None:
        config.training.w_q = args.w_q
    if args.w_c is not None:
        config.training.w_c = args.w_c
    if args.mixed_precision:
        config.training.mixed_precision = True
    if args.save_dir:
        config.training.save_dir = args.save_dir
    
    # 可选功能
    if args.use_refinement:
        config.model.use_refinement = True
    if args.use_explanations:
        config.model.use_explanations = True
    
    return config


if __name__ == "__main__":
    # 测试配置系统
    print("🧪 测试配置系统\n")
    
    # 创建默认配置
    config = Config()
    config.experiment_name = "test_config"
    config.notes = "这是一个测试配置"
    
    # 打印配置
    config.print_summary()
    
    # 保存配置
    config.save("test_config.json")
    
    # 加载配置
    loaded_config = Config.load("test_config.json")
    loaded_config.print_summary()
    
    print("✅ 配置系统测试通过！")
