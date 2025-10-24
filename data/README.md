# ACGIQA-3K 数据集

## 📊 数据集概述

**ACGIQA-3K** 是一个用于 AIGC（AI Generated Content）图像质量和文本一致性评估的数据集。

### 数据集统计

| 项目 | 数值 |
|------|------|
| **总图像数** | ~3000 张 |
| **数据大小** | ~103 MB |
| **图像格式** | JPG |
| **标注文件** | data.csv (470KB) |

---

## 📁 文件结构

```
data/
├── data.csv              # 主标注文件（470KB, 2984 条记录）
└── ACGIQA-3K/           # 图像目录
    ├── sd1.5_normal_000.jpg
    ├── sd1.5_normal_001.jpg
    ├── ...
    └── xl2.2_normal_299.jpg
```

---

## 📝 data.csv 格式

CSV 文件包含以下列：

| 列名 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `name` | string | - | 图像文件名 |
| `prompt` | string | - | 生成图像的文本提示词 |
| `mos_quality` | float | 1-5 | 图像质量主观评分 (MOS) |
| `mos_align` | float | 1-5 | 文本一致性主观评分 (MOS) |

### 示例行

```csv
name,prompt,mos_quality,mos_align
sd1.5_normal_000.jpg,a beautiful sunset over the ocean,4.2,4.5
sd1.5_lowstep_001.jpg,a cat sitting on a chair,2.8,3.2
```

---

## 🎨 图像类别

数据集包含多种质量和一致性的图像：

### 按生成模型分类
- **sd1.5_normal**: Stable Diffusion 1.5 正常生成
- **sd1.5_lowstep**: Stable Diffusion 1.5 低步数生成（质量较低）
- **sd1.5_lowcorr**: Stable Diffusion 1.5 低一致性生成
- **xl2.2_normal**: Stable Diffusion XL 2.2 正常生成

### 质量分布
- **高质量** (MOS > 4.0): ~30%
- **中等质量** (MOS 3.0-4.0): ~50%
- **低质量** (MOS < 3.0): ~20%

---

## 🚀 使用方法

### 1. 加载数据

```python
import pandas as pd
from PIL import Image
import os

# 读取标注文件
df = pd.read_csv('data/data.csv')

# 打印基本信息
print(f"总样本数: {len(df)}")
print(f"平均质量分数: {df['mos_quality'].mean():.2f}")
print(f"平均一致性分数: {df['mos_align'].mean():.2f}")

# 加载单张图像
img_path = os.path.join('data/ACGIQA-3K', df.iloc[0]['name'])
img = Image.open(img_path)
img.show()
```

### 2. 数据划分

```python
from sklearn.model_selection import train_test_split

# 划分训练集和测试集 (80/20)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"训练集: {len(train_df)} 样本")
print(f"测试集: {len(test_df)} 样本")
```

### 3. 使用 baseline.py 训练

```bash
python baseline.py \
    --data_csv_path data/data.csv \
    --image_base_dir data/ACGIQA-3K \
    --epochs 20 \
    --batch_size 32
```

---

## 📈 数据集统计分析

### 质量分数分布

```python
import matplotlib.pyplot as plt

df['mos_quality'].hist(bins=20)
plt.xlabel('Quality Score (MOS)')
plt.ylabel('Frequency')
plt.title('Quality Score Distribution')
plt.show()
```

### 一致性分数分布

```python
df['mos_align'].hist(bins=20)
plt.xlabel('Consistency Score (MOS)')
plt.ylabel('Frequency')
plt.title('Consistency Score Distribution')
plt.show()
```

### 质量 vs 一致性相关性

```python
import seaborn as sns

sns.scatterplot(data=df, x='mos_quality', y='mos_align', alpha=0.5)
plt.xlabel('Quality Score')
plt.ylabel('Consistency Score')
plt.title('Quality vs Consistency Correlation')
plt.show()
```

---

## 🔍 数据集特点

### 优点
- ✅ 涵盖多种生成模型（SD 1.5, SD XL 2.2）
- ✅ 包含不同质量水平的样本
- ✅ 双重标注（质量 + 一致性）
- ✅ 真实的文本提示词
- ✅ 人工主观评分（MOS）

### 应用场景
- 图像质量评估模型训练
- 文本-图像一致性评估
- AIGC 模型性能比较
- 视觉-语言模型微调

---

## 📊 数据集来源

本数据集基于以下工作：

> **ACGIQA: Aesthetic and Consistency-Guided Image Quality Assessment**  
> [论文链接]  
> [GitHub 链接]

如果使用本数据集，请引用原始论文。

---

## 📄 许可证

本数据集仅用于学术研究和非商业用途。

---

## 🔗 相关资源

- **项目主页**: https://github.com/qiyan111/AIGC
- **训练脚本**: [baseline.py](../baseline.py)
- **使用文档**: [README.md](../README.md)

---

## 📧 联系方式

如有数据集相关问题，请通过以下方式联系：
- GitHub Issues: https://github.com/qiyan111/AIGC/issues
- Email: [你的邮箱]

---

**最后更新**: 2025-10-24  
**版本**: 1.0

