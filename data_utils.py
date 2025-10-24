#!/usr/bin/env python3
"""
数据预处理和分析工具
用于数据集统计、可视化和预处理
"""

import argparse
import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")


class DatasetAnalyzer:
    """数据集分析器"""
    
    def __init__(self, csv_path: str, image_dir: str, output_dir: str = "data_analysis"):
        """
        初始化分析器
        
        Args:
            csv_path: CSV 文件路径
            image_dir: 图像目录
            output_dir: 输出目录
        """
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"📂 加载数据集: {csv_path}")
        print(f"📊 样本数: {len(self.df)}")
    
    def print_basic_stats(self):
        """打印基本统计信息"""
        print("\n" + "=" * 80)
        print("📊 数据集基本统计")
        print("=" * 80)
        
        print(f"\n样本总数: {len(self.df)}")
        
        if 'mos_quality' in self.df.columns:
            print(f"\n🎨 Quality 分数:")
            print(f"  • 均值: {self.df['mos_quality'].mean():.3f}")
            print(f"  • 标准差: {self.df['mos_quality'].std():.3f}")
            print(f"  • 最小值: {self.df['mos_quality'].min():.3f}")
            print(f"  • 最大值: {self.df['mos_quality'].max():.3f}")
            print(f"  • 中位数: {self.df['mos_quality'].median():.3f}")
        
        if 'mos_align' in self.df.columns:
            print(f"\n🔗 Consistency 分数:")
            print(f"  • 均值: {self.df['mos_align'].mean():.3f}")
            print(f"  • 标准差: {self.df['mos_align'].std():.3f}")
            print(f"  • 最小值: {self.df['mos_align'].min():.3f}")
            print(f"  • 最大值: {self.df['mos_align'].max():.3f}")
            print(f"  • 中位数: {self.df['mos_align'].median():.3f}")
        
        if 'prompt' in self.df.columns:
            prompt_lengths = self.df['prompt'].str.len()
            print(f"\n📝 Prompt 统计:")
            print(f"  • 平均长度: {prompt_lengths.mean():.1f} 字符")
            print(f"  • 最短: {prompt_lengths.min()} 字符")
            print(f"  • 最长: {prompt_lengths.max()} 字符")
        
        print("=" * 80 + "\n")
    
    def plot_score_distributions(self, save_path: Optional[str] = None):
        """绘制分数分布图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Quality 直方图
        if 'mos_quality' in self.df.columns:
            ax = axes[0, 0]
            ax.hist(self.df['mos_quality'], bins=30, color='skyblue',
                   edgecolor='black', alpha=0.7)
            ax.axvline(self.df['mos_quality'].mean(), color='red',
                      linestyle='--', linewidth=2,
                      label=f'均值 ({self.df["mos_quality"].mean():.2f})')
            ax.set_xlabel('Quality 分数', fontsize=12)
            ax.set_ylabel('频数', fontsize=12)
            ax.set_title('Quality 分数分布', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Quality 箱线图
        if 'mos_quality' in self.df.columns:
            ax = axes[0, 1]
            ax.boxplot([self.df['mos_quality']], vert=True, labels=['Quality'],
                      patch_artist=True, boxprops=dict(facecolor='skyblue'))
            ax.set_ylabel('分数', fontsize=12)
            ax.set_title('Quality 分数箱线图', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Consistency 直方图
        if 'mos_align' in self.df.columns:
            ax = axes[1, 0]
            ax.hist(self.df['mos_align'], bins=30, color='orange',
                   edgecolor='black', alpha=0.7)
            ax.axvline(self.df['mos_align'].mean(), color='red',
                      linestyle='--', linewidth=2,
                      label=f'均值 ({self.df["mos_align"].mean():.2f})')
            ax.set_xlabel('Consistency 分数', fontsize=12)
            ax.set_ylabel('频数', fontsize=12)
            ax.set_title('Consistency 分数分布', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Consistency 箱线图
        if 'mos_align' in self.df.columns:
            ax = axes[1, 1]
            ax.boxplot([self.df['mos_align']], vert=True, labels=['Consistency'],
                      patch_artist=True, boxprops=dict(facecolor='orange'))
            ax.set_ylabel('分数', fontsize=12)
            ax.set_title('Consistency 分数箱线图', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'score_distributions.png'),
                       dpi=300, bbox_inches='tight')
        
        plt.close()
        print("💾 分数分布图已保存")
    
    def plot_score_correlation(self, save_path: Optional[str] = None):
        """绘制 Quality 与 Consistency 相关性图"""
        if 'mos_quality' not in self.df.columns or 'mos_align' not in self.df.columns:
            print("⚠️  缺少必要的列，无法绘制相关性图")
            return
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(self.df['mos_quality'], self.df['mos_align'],
                  alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
        
        # 计算相关系数
        corr = self.df['mos_quality'].corr(self.df['mos_align'])
        
        ax.set_xlabel('Quality 分数', fontsize=12)
        ax.set_ylabel('Consistency 分数', fontsize=12)
        ax.set_title(f'Quality vs Consistency 相关性\n相关系数: {corr:.3f}',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, 'score_correlation.png'),
                       dpi=300, bbox_inches='tight')
        
        plt.close()
        print("💾 相关性图已保存")
    
    def check_data_quality(self):
        """检查数据质量（缺失值、异常值等）"""
        print("\n" + "=" * 80)
        print("🔍 数据质量检查")
        print("=" * 80 + "\n")
        
        # 检查缺失值
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            print("⚠️  发现缺失值:")
            for col, count in missing[missing > 0].items():
                print(f"  • {col}: {count} ({count/len(self.df)*100:.1f}%)")
        else:
            print("✅ 无缺失值")
        
        # 检查重复行
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"\n⚠️  发现 {duplicates} 个重复行")
        else:
            print("\n✅ 无重复行")
        
        # 检查分数范围
        print("\n📊 分数范围检查:")
        if 'mos_quality' in self.df.columns:
            q_min, q_max = self.df['mos_quality'].min(), self.df['mos_quality'].max()
            if q_min < 1 or q_max > 5:
                print(f"  ⚠️  Quality 分数超出范围 [1, 5]: [{q_min:.2f}, {q_max:.2f}]")
            else:
                print(f"  ✅ Quality 分数在正常范围: [{q_min:.2f}, {q_max:.2f}]")
        
        if 'mos_align' in self.df.columns:
            c_min, c_max = self.df['mos_align'].min(), self.df['mos_align'].max()
            if c_min < 1 or c_max > 5:
                print(f"  ⚠️  Consistency 分数超出范围 [1, 5]: [{c_min:.2f}, {c_max:.2f}]")
            else:
                print(f"  ✅ Consistency 分数在正常范围: [{c_min:.2f}, {c_max:.2f}]")
        
        # 检查图像文件是否存在
        print("\n🖼️  图像文件检查:")
        missing_images = []
        for img_name in tqdm(self.df['name'], desc="检查图像文件"):
            img_path = os.path.join(self.image_dir, img_name)
            if not os.path.exists(img_path):
                missing_images.append(img_name)
        
        if missing_images:
            print(f"  ⚠️  发现 {len(missing_images)} 个缺失的图像文件")
            print(f"  前 5 个: {missing_images[:5]}")
        else:
            print(f"  ✅ 所有图像文件都存在")
        
        print("=" * 80 + "\n")
    
    def analyze_image_properties(self, sample_size: int = 100):
        """分析图像属性（尺寸、格式等）"""
        print("\n" + "=" * 80)
        print(f"🖼️  图像属性分析 (采样 {sample_size} 张)")
        print("=" * 80 + "\n")
        
        # 随机采样
        sample_df = self.df.sample(min(sample_size, len(self.df)))
        
        widths, heights, sizes = [], [], []
        formats = {}
        
        for img_name in tqdm(sample_df['name'], desc="分析图像"):
            img_path = os.path.join(self.image_dir, img_name)
            try:
                img = Image.open(img_path)
                widths.append(img.width)
                heights.append(img.height)
                sizes.append(os.path.getsize(img_path) / 1024)  # KB
                
                fmt = img.format
                formats[fmt] = formats.get(fmt, 0) + 1
            except Exception as e:
                print(f"⚠️  无法读取 {img_name}: {e}")
        
        print(f"📐 图像尺寸:")
        print(f"  • 宽度: 均值={np.mean(widths):.0f}, 范围=[{min(widths)}, {max(widths)}]")
        print(f"  • 高度: 均值={np.mean(heights):.0f}, 范围=[{min(heights)}, {max(heights)}]")
        
        print(f"\n💾 文件大小:")
        print(f"  • 均值: {np.mean(sizes):.1f} KB")
        print(f"  • 范围: [{min(sizes):.1f}, {max(sizes):.1f}] KB")
        
        print(f"\n🎨 图像格式:")
        for fmt, count in formats.items():
            print(f"  • {fmt}: {count} ({count/len(widths)*100:.1f}%)")
        
        print("=" * 80 + "\n")
    
    def run_full_analysis(self):
        """运行完整的数据分析"""
        print("\n" + "=" * 80)
        print("🚀 开始完整数据分析")
        print("=" * 80)
        
        # 1. 基本统计
        self.print_basic_stats()
        
        # 2. 数据质量检查
        self.check_data_quality()
        
        # 3. 图像属性分析
        self.analyze_image_properties()
        
        # 4. 可视化
        print("📊 生成可视化图表...")
        self.plot_score_distributions()
        self.plot_score_correlation()
        
        print("\n✅ 数据分析完成！结果已保存到:", self.output_dir)


class DataPreprocessor:
    """数据预处理器"""
    
    def __init__(self, csv_path: str):
        """
        初始化预处理器
        
        Args:
            csv_path: CSV 文件路径
        """
        self.df = pd.read_csv(csv_path)
        print(f"📂 加载数据: {csv_path}")
        print(f"📊 样本数: {len(self.df)}")
    
    def remove_outliers(self, column: str, n_std: float = 3.0) -> pd.DataFrame:
        """
        移除异常值（基于标准差）
        
        Args:
            column: 列名
            n_std: 标准差倍数
            
        Returns:
            处理后的 DataFrame
        """
        mean = self.df[column].mean()
        std = self.df[column].std()
        
        lower_bound = mean - n_std * std
        upper_bound = mean + n_std * std
        
        outliers = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        n_outliers = outliers.sum()
        
        print(f"🔍 {column} 列:")
        print(f"  • 检测到 {n_outliers} 个异常值 (±{n_std}σ)")
        print(f"  • 范围: [{lower_bound:.3f}, {upper_bound:.3f}]")
        
        self.df = self.df[~outliers]
        return self.df
    
    def normalize_scores(self, columns: list, target_range: Tuple[float, float] = (0, 1)):
        """
        归一化分数到目标范围
        
        Args:
            columns: 要归一化的列名列表
            target_range: 目标范围 (min, max)
        """
        for col in columns:
            if col not in self.df.columns:
                continue
            
            min_val, max_val = self.df[col].min(), self.df[col].max()
            target_min, target_max = target_range
            
            self.df[col] = (self.df[col] - min_val) / (max_val - min_val) * \
                          (target_max - target_min) + target_min
            
            print(f"✅ {col} 已归一化到 [{target_min}, {target_max}]")
    
    def balance_dataset(self, column: str, n_bins: int = 5) -> pd.DataFrame:
        """
        平衡数据集（使各分数区间样本数相近）
        
        Args:
            column: 用于分箱的列名
            n_bins: 分箱数
            
        Returns:
            平衡后的 DataFrame
        """
        # 分箱
        self.df['bin'] = pd.cut(self.df[column], bins=n_bins, labels=False)
        
        # 计算每个箱的样本数
        bin_counts = self.df['bin'].value_counts().sort_index()
        min_count = bin_counts.min()
        
        print(f"\n📊 分箱统计 ({column}):")
        for i, count in enumerate(bin_counts):
            print(f"  Bin {i}: {count} 样本")
        
        # 下采样到最小数量
        balanced_df = []
        for i in range(n_bins):
            bin_data = self.df[self.df['bin'] == i]
            sampled = bin_data.sample(min(len(bin_data), min_count), random_state=42)
            balanced_df.append(sampled)
        
        self.df = pd.concat(balanced_df, ignore_index=True)
        self.df = self.df.drop('bin', axis=1)
        
        print(f"\n✅ 数据集已平衡，总样本数: {len(self.df)}")
        return self.df
    
    def save(self, output_path: str):
        """保存处理后的数据"""
        self.df.to_csv(output_path, index=False)
        print(f"💾 处理后的数据已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="数据集分析和预处理工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # 分析命令
    analyze_parser = subparsers.add_parser('analyze', help='数据集分析')
    analyze_parser.add_argument('--csv', type=str, required=True, help='CSV 文件路径')
    analyze_parser.add_argument('--image_dir', type=str, required=True, help='图像目录')
    analyze_parser.add_argument('--output_dir', type=str, default='data_analysis',
                               help='输出目录')
    
    # 预处理命令
    preprocess_parser = subparsers.add_parser('preprocess', help='数据预处理')
    preprocess_parser.add_argument('--csv', type=str, required=True, help='CSV 文件路径')
    preprocess_parser.add_argument('--output', type=str, required=True, help='输出文件路径')
    preprocess_parser.add_argument('--remove_outliers', action='store_true',
                                   help='移除异常值')
    preprocess_parser.add_argument('--balance', action='store_true',
                                   help='平衡数据集')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        # 数据分析
        analyzer = DatasetAnalyzer(args.csv, args.image_dir, args.output_dir)
        analyzer.run_full_analysis()
    
    elif args.command == 'preprocess':
        # 数据预处理
        preprocessor = DataPreprocessor(args.csv)
        
        if args.remove_outliers:
            print("\n🔧 移除异常值...")
            if 'mos_quality' in preprocessor.df.columns:
                preprocessor.remove_outliers('mos_quality')
            if 'mos_align' in preprocessor.df.columns:
                preprocessor.remove_outliers('mos_align')
        
        if args.balance:
            print("\n⚖️  平衡数据集...")
            if 'mos_quality' in preprocessor.df.columns:
                preprocessor.balance_dataset('mos_quality')
        
        preprocessor.save(args.output)
        print("\n✅ 预处理完成！")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
