#!/usr/bin/env python3
"""
评估和可视化脚本
用于全面评估模型性能并生成可视化报告
"""

import argparse
import os
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")


class ModelEvaluator:
    """模型评估器 - 计算各种评估指标并生成可视化"""
    
    def __init__(self, predictions_csv: str, output_dir: str = "evaluation_results"):
        """
        初始化评估器
        
        Args:
            predictions_csv: 包含预测结果的 CSV 文件（需包含 gt 和 pred 列）
            output_dir: 输出目录
        """
        self.df = pd.read_csv(predictions_csv)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📂 加载预测结果: {predictions_csv}")
        print(f"📊 样本数: {len(self.df)}")
    
    def compute_metrics(self) -> Dict[str, float]:
        """计算所有评估指标"""
        metrics = {}
        
        # Quality 指标
        if 'mos_quality' in self.df.columns and 'quality_score' in self.df.columns:
            q_gt = self.df['mos_quality'].values
            q_pred = self.df['quality_score'].values
            
            # 过滤 NaN 值
            valid_mask = ~(np.isnan(q_gt) | np.isnan(q_pred))
            q_gt = q_gt[valid_mask]
            q_pred = q_pred[valid_mask]
            
            metrics['quality_srocc'] = spearmanr(q_gt, q_pred).correlation
            metrics['quality_plcc'] = pearsonr(q_gt, q_pred)[0]
            metrics['quality_mae'] = mean_absolute_error(q_gt, q_pred)
            metrics['quality_rmse'] = np.sqrt(mean_squared_error(q_gt, q_pred))
        
        # Consistency 指标
        if 'mos_align' in self.df.columns and 'consistency_score' in self.df.columns:
            c_gt = self.df['mos_align'].values
            c_pred = self.df['consistency_score'].values
            
            # 过滤 NaN 值
            valid_mask = ~(np.isnan(c_gt) | np.isnan(c_pred))
            c_gt = c_gt[valid_mask]
            c_pred = c_pred[valid_mask]
            
            metrics['consistency_srocc'] = spearmanr(c_gt, c_pred).correlation
            metrics['consistency_plcc'] = pearsonr(c_gt, c_pred)[0]
            metrics['consistency_mae'] = mean_absolute_error(c_gt, c_pred)
            metrics['consistency_rmse'] = np.sqrt(mean_squared_error(c_gt, c_pred))
        
        return metrics
    
    def plot_scatter(self, save_path: str = None):
        """绘制预测值 vs 真实值散点图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Quality 散点图
        if 'mos_quality' in self.df.columns and 'quality_score' in self.df.columns:
            ax = axes[0]
            q_gt = self.df['mos_quality']
            q_pred = self.df['quality_score']
            
            ax.scatter(q_gt, q_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
            ax.plot([1, 5], [1, 5], 'r--', lw=2, label='理想预测')
            
            # 计算并显示指标
            srocc = spearmanr(q_gt, q_pred).correlation
            plcc = pearsonr(q_gt, q_pred)[0]
            mae = mean_absolute_error(q_gt, q_pred)
            
            ax.set_xlabel('真实质量分数 (Ground Truth)', fontsize=12)
            ax.set_ylabel('预测质量分数 (Predicted)', fontsize=12)
            ax.set_title(f'Quality 预测\nSROCC={srocc:.4f}, PLCC={plcc:.4f}, MAE={mae:.3f}',
                        fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)
        
        # Consistency 散点图
        if 'mos_align' in self.df.columns and 'consistency_score' in self.df.columns:
            ax = axes[1]
            c_gt = self.df['mos_align']
            c_pred = self.df['consistency_score']
            
            ax.scatter(c_gt, c_pred, alpha=0.5, s=30, edgecolors='k', linewidth=0.5,
                      color='orange')
            ax.plot([1, 5], [1, 5], 'r--', lw=2, label='理想预测')
            
            # 计算并显示指标
            srocc = spearmanr(c_gt, c_pred).correlation
            plcc = pearsonr(c_gt, c_pred)[0]
            mae = mean_absolute_error(c_gt, c_pred)
            
            ax.set_xlabel('真实一致性分数 (Ground Truth)', fontsize=12)
            ax.set_ylabel('预测一致性分数 (Predicted)', fontsize=12)
            ax.set_title(f'Consistency 预测\nSROCC={srocc:.4f}, PLCC={plcc:.4f}, MAE={mae:.3f}',
                        fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 散点图已保存: {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'scatter_plot.png'),
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_error_distribution(self, save_path: str = None):
        """绘制误差分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Quality 误差分布
        if 'mos_quality' in self.df.columns and 'quality_score' in self.df.columns:
            q_error = self.df['quality_score'] - self.df['mos_quality']
            
            # 直方图
            ax = axes[0, 0]
            ax.hist(q_error, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差')
            ax.axvline(q_error.mean(), color='green', linestyle='--', linewidth=2,
                      label=f'平均误差 ({q_error.mean():.3f})')
            ax.set_xlabel('预测误差 (Predicted - Ground Truth)', fontsize=11)
            ax.set_ylabel('频数', fontsize=11)
            ax.set_title(f'Quality 误差分布\nStd={q_error.std():.3f}',
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 箱线图
            ax = axes[0, 1]
            ax.boxplot([q_error], vert=True, labels=['Quality'], patch_artist=True,
                      boxprops=dict(facecolor='skyblue'))
            ax.axhline(0, color='red', linestyle='--', linewidth=2)
            ax.set_ylabel('预测误差', fontsize=11)
            ax.set_title('Quality 误差箱线图', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Consistency 误差分布
        if 'mos_align' in self.df.columns and 'consistency_score' in self.df.columns:
            c_error = self.df['consistency_score'] - self.df['mos_align']
            
            # 直方图
            ax = axes[1, 0]
            ax.hist(c_error, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax.axvline(0, color='red', linestyle='--', linewidth=2, label='零误差')
            ax.axvline(c_error.mean(), color='green', linestyle='--', linewidth=2,
                      label=f'平均误差 ({c_error.mean():.3f})')
            ax.set_xlabel('预测误差 (Predicted - Ground Truth)', fontsize=11)
            ax.set_ylabel('频数', fontsize=11)
            ax.set_title(f'Consistency 误差分布\nStd={c_error.std():.3f}',
                        fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 箱线图
            ax = axes[1, 1]
            ax.boxplot([c_error], vert=True, labels=['Consistency'], patch_artist=True,
                      boxprops=dict(facecolor='orange'))
            ax.axhline(0, color='red', linestyle='--', linewidth=2)
            ax.set_ylabel('预测误差', fontsize=11)
            ax.set_title('Consistency 误差箱线图', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 误差分布图已保存: {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'error_distribution.png'),
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def plot_score_distribution(self, save_path: str = None):
        """绘制分数分布对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Quality 分数分布
        if 'mos_quality' in self.df.columns and 'quality_score' in self.df.columns:
            ax = axes[0]
            ax.hist(self.df['mos_quality'], bins=30, alpha=0.6, label='真实值',
                   color='blue', edgecolor='black')
            ax.hist(self.df['quality_score'], bins=30, alpha=0.6, label='预测值',
                   color='red', edgecolor='black')
            ax.set_xlabel('Quality 分数', fontsize=12)
            ax.set_ylabel('频数', fontsize=12)
            ax.set_title('Quality 分数分布对比', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Consistency 分数分布
        if 'mos_align' in self.df.columns and 'consistency_score' in self.df.columns:
            ax = axes[1]
            ax.hist(self.df['mos_align'], bins=30, alpha=0.6, label='真实值',
                   color='blue', edgecolor='black')
            ax.hist(self.df['consistency_score'], bins=30, alpha=0.6, label='预测值',
                   color='red', edgecolor='black')
            ax.set_xlabel('Consistency 分数', fontsize=12)
            ax.set_ylabel('频数', fontsize=12)
            ax.set_title('Consistency 分数分布对比', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 分数分布图已保存: {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, 'score_distribution.png'),
                       dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_report(self, save_path: str = None):
        """生成完整的评估报告（文本）"""
        metrics = self.compute_metrics()
        
        report_lines = [
            "=" * 80,
            "📊 模型评估报告",
            "=" * 80,
            "",
            f"📂 数据集: {len(self.df)} 个样本",
            "",
            "🎯 Quality 评估指标:",
            "-" * 80,
        ]
        
        if 'quality_srocc' in metrics:
            report_lines.extend([
                f"  • SROCC (Spearman):    {metrics['quality_srocc']:.4f}",
                f"  • PLCC (Pearson):      {metrics['quality_plcc']:.4f}",
                f"  • MAE:                 {metrics['quality_mae']:.4f}",
                f"  • RMSE:                {metrics['quality_rmse']:.4f}",
            ])
        
        report_lines.extend([
            "",
            "🔗 Consistency 评估指标:",
            "-" * 80,
        ])
        
        if 'consistency_srocc' in metrics:
            report_lines.extend([
                f"  • SROCC (Spearman):    {metrics['consistency_srocc']:.4f}",
                f"  • PLCC (Pearson):      {metrics['consistency_plcc']:.4f}",
                f"  • MAE:                 {metrics['consistency_mae']:.4f}",
                f"  • RMSE:                {metrics['consistency_rmse']:.4f}",
            ])
        
        report_lines.extend([
            "",
            "=" * 80,
            "📈 性能等级评估:",
            "-" * 80,
        ])
        
        # 性能评级
        if 'quality_srocc' in metrics:
            q_srocc = metrics['quality_srocc']
            q_grade = "优秀" if q_srocc > 0.90 else "良好" if q_srocc > 0.85 else "及格" if q_srocc > 0.80 else "较差"
            report_lines.append(f"  • Quality SROCC:       {q_grade} ({q_srocc:.4f})")
        
        if 'consistency_srocc' in metrics:
            c_srocc = metrics['consistency_srocc']
            c_grade = "优秀" if c_srocc > 0.90 else "良好" if c_srocc > 0.85 else "及格" if c_srocc > 0.80 else "较差"
            report_lines.append(f"  • Consistency SROCC:   {c_grade} ({c_srocc:.4f})")
        
        report_lines.extend([
            "",
            "=" * 80,
            "💡 建议:",
            "-" * 80,
        ])
        
        # 给出改进建议
        if 'quality_srocc' in metrics and metrics['quality_srocc'] < 0.85:
            report_lines.append("  ⚠️  Quality 性能较低，建议：")
            report_lines.append("     - 增大 --residual_scale_q 参数")
            report_lines.append("     - 增大 --w_q 损失权重")
            report_lines.append("     - 减少冻结层数（更多层可训练）")
        
        if 'consistency_srocc' in metrics and metrics['consistency_srocc'] < 0.85:
            report_lines.append("  ⚠️  Consistency 性能较低，建议：")
            report_lines.append("     - 增大 --residual_scale_c 参数")
            report_lines.append("     - 增大 --w_c 损失权重")
            report_lines.append("     - 考虑使用 --use_refinement 选项")
        
        if 'quality_srocc' in metrics and metrics['quality_srocc'] >= 0.90 and \
           'consistency_srocc' in metrics and metrics['consistency_srocc'] >= 0.90:
            report_lines.append("  ✅ 模型性能优秀！可以考虑：")
            report_lines.append("     - 在更大数据集上测试泛化能力")
            report_lines.append("     - 导出为 ONNX 用于部署")
            report_lines.append("     - 尝试模型蒸馏以减小模型大小")
        
        report_lines.append("=" * 80)
        
        report = "\n".join(report_lines)
        
        # 打印到控制台
        print(report)
        
        # 保存到文件
        if save_path:
            report_path = save_path
        else:
            report_path = os.path.join(self.output_dir, 'evaluation_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n💾 评估报告已保存: {report_path}")
        
        return metrics
    
    def run_full_evaluation(self):
        """运行完整的评估流程"""
        print("\n" + "=" * 80)
        print("🚀 开始完整评估...")
        print("=" * 80 + "\n")
        
        # 1. 计算指标
        print("1️⃣  计算评估指标...")
        metrics = self.compute_metrics()
        
        # 2. 生成散点图
        print("2️⃣  生成散点图...")
        self.plot_scatter()
        
        # 3. 生成误差分布图
        print("3️⃣  生成误差分布图...")
        self.plot_error_distribution()
        
        # 4. 生成分数分布图
        print("4️⃣  生成分数分布图...")
        self.plot_score_distribution()
        
        # 5. 生成评估报告
        print("5️⃣  生成评估报告...\n")
        self.generate_report()
        
        print("\n✅ 评估完成！所有结果已保存到:", self.output_dir)
        
        return metrics


def main():
    parser = argparse.ArgumentParser(
        description="模型评估和可视化工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--predictions_csv', type=str, required=True,
                       help='包含预测结果的 CSV 文件')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='输出目录')
    parser.add_argument('--plot_only', action='store_true',
                       help='仅生成图表，不计算指标')
    
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = ModelEvaluator(args.predictions_csv, args.output_dir)
    
    if args.plot_only:
        print("📊 生成可视化图表...")
        evaluator.plot_scatter()
        evaluator.plot_error_distribution()
        evaluator.plot_score_distribution()
        print("✅ 图表生成完成！")
    else:
        # 运行完整评估
        evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
