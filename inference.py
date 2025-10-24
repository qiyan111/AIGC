#!/usr/bin/env python3
"""
推理脚本 - 使用训练好的模型对图像进行质量和一致性评估
支持单张图像和批量图像评估
"""

import argparse
import os
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor
from torchvision import transforms
import pandas as pd
from tqdm import tqdm

# 导入模型定义
from baseline import BaselineCLIPScore


class ImageQualityPredictor:
    """图像质量和一致性预测器"""
    
    def __init__(
        self,
        model_path: str,
        clip_model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        use_residual_learning: bool = True,
        partial_freeze: bool = False,
        freeze_layers: int = 18,
        dropout: float = 0.1
    ):
        """
        初始化预测器
        
        Args:
            model_path: 训练好的模型权重路径
            clip_model_name: CLIP 模型名称
            device: 设备 (cuda/cpu)
            use_residual_learning: 是否使用残差学习
            partial_freeze: 是否部分冻结
            freeze_layers: 冻结层数
            dropout: Dropout 比例
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"🔧 使用设备: {self.device}")
        
        # 加载 CLIP 处理器
        print(f"📦 加载 CLIP 处理器: {clip_model_name}")
        self.processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        # 创建图像变换
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                self.processor.image_processor.image_mean,
                self.processor.image_processor.image_std
            )
        ])
        
        # 加载模型
        print(f"🤖 加载模型: {model_path}")
        self.model = BaselineCLIPScore(
            clip_model_name=clip_model_name,
            freeze=False,
            use_residual_learning=use_residual_learning,
            residual_scale_q=0.2,
            residual_scale_c=0.2,
            partial_freeze=partial_freeze,
            freeze_layers=freeze_layers,
            dropout=dropout
        ).to(self.device)
        
        # 加载权重
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("✅ 模型加载完成\n")
    
    @torch.no_grad()
    def predict_single(
        self,
        image_path: str,
        prompt: str,
        return_details: bool = False
    ) -> Tuple[float, float]:
        """
        预测单张图像的质量和一致性
        
        Args:
            image_path: 图像路径
            prompt: 文本提示词
            return_details: 是否返回详细信息（包括 coarse score）
            
        Returns:
            (quality_score, consistency_score) 或
            (quality_score, consistency_score, c_coarse_score) 如果 return_details=True
        """
        # 加载图像
        image = Image.open(image_path).convert("RGB")
        
        # 预处理图像
        pixel_values = self.transform(image).unsqueeze(0).to(self.device)
        
        # 处理文本
        text_inputs = self.processor(
            text=[prompt],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        # 推理
        q_pred, c_pred, _, _, c_coarse = self.model(pixel_values, input_ids, attention_mask)
        
        # 转换为 1-5 分制
        quality_score = q_pred.item() * 5.0
        consistency_score = c_pred.item() * 5.0
        
        if return_details:
            c_coarse_score = c_coarse.item() * 5.0
            return quality_score, consistency_score, c_coarse_score
        
        return quality_score, consistency_score
    
    def predict_batch(
        self,
        image_paths: List[str],
        prompts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        批量预测图像质量和一致性
        
        Args:
            image_paths: 图像路径列表
            prompts: 文本提示词列表
            batch_size: 批次大小
            show_progress: 是否显示进度条
            
        Returns:
            包含预测结果的 DataFrame
        """
        assert len(image_paths) == len(prompts), "图像数量和提示词数量必须相同"
        
        results = []
        iterator = range(0, len(image_paths), batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, desc="预测进度")
        
        for i in iterator:
            batch_images = image_paths[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            
            # 批量预处理图像
            pixel_values_list = []
            for img_path in batch_images:
                try:
                    image = Image.open(img_path).convert("RGB")
                    pixel_values = self.transform(image)
                    pixel_values_list.append(pixel_values)
                except Exception as e:
                    print(f"⚠️  加载图像失败: {img_path}, 错误: {e}")
                    continue
            
            if not pixel_values_list:
                continue
            
            pixel_values = torch.stack(pixel_values_list).to(self.device)
            
            # 批量处理文本
            text_inputs = self.processor(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            )
            input_ids = text_inputs.input_ids.to(self.device)
            attention_mask = text_inputs.attention_mask.to(self.device)
            
            # 推理
            with torch.no_grad():
                q_pred, c_pred, _, _, _ = self.model(pixel_values, input_ids, attention_mask)
            
            # 收集结果
            for j, (img_path, prompt) in enumerate(zip(batch_images, batch_prompts)):
                results.append({
                    'image_path': img_path,
                    'prompt': prompt,
                    'quality_score': q_pred[j].item() * 5.0,
                    'consistency_score': c_pred[j].item() * 5.0
                })
        
        return pd.DataFrame(results)
    
    def predict_from_csv(
        self,
        csv_path: str,
        image_dir: str,
        output_path: Optional[str] = None,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        从 CSV 文件读取数据并批量预测
        
        Args:
            csv_path: CSV 文件路径（需包含 'name' 和 'prompt' 列）
            image_dir: 图像目录
            output_path: 输出 CSV 路径（可选）
            batch_size: 批次大小
            
        Returns:
            包含预测结果的 DataFrame
        """
        print(f"📂 读取 CSV 文件: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 检查必需列
        if 'name' not in df.columns or 'prompt' not in df.columns:
            raise ValueError("CSV 文件必须包含 'name' 和 'prompt' 列")
        
        # 构建图像路径
        image_paths = [os.path.join(image_dir, name) for name in df['name']]
        prompts = df['prompt'].tolist()
        
        # 批量预测
        print(f"🚀 开始预测 {len(image_paths)} 张图像...")
        results_df = self.predict_batch(image_paths, prompts, batch_size=batch_size)
        
        # 合并原始数据和预测结果
        results_df = pd.merge(
            df,
            results_df[['image_path', 'quality_score', 'consistency_score']],
            left_on=df['name'].apply(lambda x: os.path.join(image_dir, x)),
            right_on='image_path',
            how='left'
        )
        
        # 如果有真实标签，计算误差
        if 'mos_quality' in df.columns:
            results_df['quality_error'] = abs(results_df['quality_score'] - results_df['mos_quality'])
        if 'mos_align' in df.columns:
            results_df['consistency_error'] = abs(results_df['consistency_score'] - results_df['mos_align'])
        
        # 保存结果
        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"💾 预测结果已保存到: {output_path}")
        
        return results_df


def main():
    parser = argparse.ArgumentParser(
        description="AIGC 图像质量和一致性评估 - 推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型权重路径')
    parser.add_argument('--clip_model_name', type=str,
                        default='openai/clip-vit-large-patch14',
                        help='CLIP 模型名称或路径')
    
    # 推理模式
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--single', nargs=2, metavar=('IMAGE', 'PROMPT'),
                           help='单张图像推理：--single image.jpg "a cat"')
    mode_group.add_argument('--batch', nargs=2, metavar=('IMAGE_DIR', 'PROMPTS_FILE'),
                           help='批量推理：--batch images/ prompts.txt')
    mode_group.add_argument('--csv', nargs=2, metavar=('CSV_FILE', 'IMAGE_DIR'),
                           help='从 CSV 推理：--csv data.csv images/')
    
    # 模型配置
    parser.add_argument('--use_residual_learning', action='store_true', default=True,
                        help='使用残差学习模式')
    parser.add_argument('--partial_freeze', action='store_true',
                        help='部分冻结模式')
    parser.add_argument('--freeze_layers', type=int, default=18,
                        help='冻结层数')
    
    # 推理设置
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批量推理的批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='推理设备')
    parser.add_argument('--output', type=str,
                        help='输出文件路径（批量推理时）')
    
    args = parser.parse_args()
    
    # 创建预测器
    print("=" * 80)
    print("🎯 AIGC 图像质量与一致性评估 - 推理模式")
    print("=" * 80 + "\n")
    
    predictor = ImageQualityPredictor(
        model_path=args.model_path,
        clip_model_name=args.clip_model_name,
        device=args.device,
        use_residual_learning=args.use_residual_learning,
        partial_freeze=args.partial_freeze,
        freeze_layers=args.freeze_layers
    )
    
    # 单张图像推理
    if args.single:
        image_path, prompt = args.single
        print(f"🖼️  图像: {image_path}")
        print(f"📝 提示词: {prompt}\n")
        
        quality, consistency, coarse = predictor.predict_single(
            image_path, prompt, return_details=True
        )
        
        print("=" * 80)
        print("📊 预测结果:")
        print("-" * 80)
        print(f"  🎨 图像质量 (Quality):         {quality:.2f} / 5.00")
        print(f"  🔗 文本一致性 (Consistency):   {consistency:.2f} / 5.00")
        if args.use_residual_learning:
            print(f"  📈 基准分数 (Coarse/Base):     {coarse:.2f} / 5.00")
            print(f"  ➕ 残差修正 (Residual):        {consistency - coarse:+.2f}")
        print("=" * 80)
    
    # 批量推理
    elif args.batch:
        image_dir, prompts_file = args.batch
        
        # 读取提示词文件
        with open(prompts_file, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        
        # 获取图像文件列表
        image_files = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
        ])
        
        if len(image_files) != len(prompts):
            print(f"⚠️  警告: 图像数量 ({len(image_files)}) 与提示词数量 ({len(prompts)}) 不匹配")
            min_len = min(len(image_files), len(prompts))
            image_files = image_files[:min_len]
            prompts = prompts[:min_len]
        
        # 批量预测
        results = predictor.predict_batch(
            image_files, prompts, batch_size=args.batch_size
        )
        
        # 显示统计信息
        print("\n" + "=" * 80)
        print("📊 预测统计:")
        print("-" * 80)
        print(f"  总图像数: {len(results)}")
        print(f"  平均质量分数:     {results['quality_score'].mean():.2f} / 5.00")
        print(f"  平均一致性分数:   {results['consistency_score'].mean():.2f} / 5.00")
        print(f"  质量分数标准差:   {results['quality_score'].std():.2f}")
        print(f"  一致性分数标准差: {results['consistency_score'].std():.2f}")
        print("=" * 80)
        
        # 保存结果
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"\n💾 结果已保存到: {args.output}")
    
    # 从 CSV 推理
    elif args.csv:
        csv_file, image_dir = args.csv
        
        results = predictor.predict_from_csv(
            csv_file, image_dir,
            output_path=args.output,
            batch_size=args.batch_size
        )
        
        # 显示统计信息
        print("\n" + "=" * 80)
        print("📊 预测统计:")
        print("-" * 80)
        print(f"  总图像数: {len(results)}")
        print(f"  平均质量分数:     {results['quality_score'].mean():.2f} / 5.00")
        print(f"  平均一致性分数:   {results['consistency_score'].mean():.2f} / 5.00")
        
        # 如果有真实标签，显示误差
        if 'quality_error' in results.columns:
            print(f"\n  📉 误差分析:")
            print(f"    质量 MAE:    {results['quality_error'].mean():.3f}")
            print(f"    一致性 MAE:  {results['consistency_error'].mean():.3f}")
        
        print("=" * 80)
    
    print("\n✅ 推理完成！")


if __name__ == "__main__":
    main()
