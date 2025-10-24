#!/usr/bin/env python3
"""baseline_clip_eval.py
Minimal AIGC image scorer with CLIP global embeddings.
Now enhanced with: AMP mixed precision, gradient accumulation, early stopping,
deterministic seeds, flexible DataLoader settings, scheduler selection, resume,
and CSV logging, while keeping residual-learning as the default.
"""

import argparse
import math
import os
import random
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import (
    CLIPModel,
    CLIPProcessor,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
import numpy as np
from datetime import datetime
import csv


class TrainingConfig:
    """Hyper-parameters with sensible defaults. Can be overridden via CLI."""

    def __init__(self):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        # Default to repository-local data; users can override via CLI
        self.data_csv_path = os.path.join(repo_root, "data", "data.csv")
        self.image_base_dir = os.path.join(repo_root, "data", "ACGIQA-3K")
        # Default to HF Hub model name; users can pass a local path instead
        self.clip_model_name = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-large-patch14")

        # training
        self.epochs = 20
        self.batch_size = 32
        self.lr = 3e-5
        self.weight_decay = 1e-4
        self.warmup_ratio = 0.05  # warmup steps 占总步数的比例（0.01-0.1）
        self.max_grad_norm = 1.0  # 梯度裁剪，0 表示不裁剪（0-2.0）
        self.dropout = 0.1  # Dropout 比例（0.0-0.3）
        self.grad_accum_steps = 1  # 梯度累积步数
        self.use_amp = True  # AMP 混合精度
        self.seed = 42  # 随机种子
        self.output_dir = os.path.join(repo_root, "outputs")
        self.log_csv = "training_log.csv"
        self.resume_from = None  # 路径：从检查点恢复
        self.early_stopping_patience = 0  # 0 表示不启用早停
        self.early_stopping_min_delta = 1e-4
        # dataloader
        self.num_workers = min(4, os.cpu_count() or 1)
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = True
        
        # loss weights
        self.w_q = 0.5
        self.w_c = 0.5
        
        # data
        self.image_size = 224
        self.test_size = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # model config
        self.freeze_clip = False  # set True for linear probing baseline
        self.use_two_branch = False
        # explanation-based distillation (optional)
        self.use_explanations = False  # enable rationale loss if CSV has 'explanation'
        self.w_exp = 0.1  # weight for rationale alignment loss
        self.explanation_column = "explanation"  # CSV column name for explanation text

        # consistency refinement module (two-stage refinement)
        self.use_refinement = False  # enable refinement module
        self.refinement_layers = 4  # number of transformer layers in refinement (2-6 recommended)
        self.refinement_heads = 8  # number of attention heads (4-8 recommended)
        self.refinement_dim = 256  # hidden dimension for refinement module
        self.strict_residual = True  # use strict residual learning (supervise residual directly)
        
        # ===== 新增：纯残差学习策略 =====
        self.use_residual_learning = True  # 启用纯残差学习（保留 CLIP 对齐空间）
        self.residual_scale_q = 0.2  # quality 残差缩放因子（限制偏离范围）
        self.residual_scale_c = 0.2  # consistency 残差缩放因子
        self.partial_freeze = False  # 部分冻结 CLIP（只训练最后几层）
        self.freeze_layers = 8  # 冻结前 N 层（ViT-L 有 24 层）

        # scheduler selection
        self.scheduler = "cosine"  # cosine | linear | constant | step
        self.step_lr_step_size = 1  # StepLR 的步长（单位：epoch）
        self.step_lr_gamma = 0.1    # StepLR 的衰减率

        # label scaling (auto by default)
        self.label_scale_q = None  # None 表示自动根据数据推断（通常 1 或 5）
        self.label_scale_c = None

        # ===== 新增：架构改进开关 =====
        # 1) 轻量级跨模态注意力，用于一致性残差特征
        self.use_cross_attn_delta = False
        self.cross_layers = 1
        self.cross_heads = 4
        self.cross_dim = 256
        self.dual_direction_cross = False  # text->image 与 image->text 双向融合

        # 1.5) 文本token ↔ 图像patch 的轻量交互（Token-Patch Cross）
        self.use_token_patch_cross = False

        # 2) 置信度门控残差（动态缩放 Δ）
        self.use_confidence_gate = False

        # 3) 质量分支使用 patch token 进行细粒度感知
        self.use_token_quality = False

        # 4) 扩展交互特征（Hadamard 与 L1 差）用于一致性残差
        self.use_ext_interactions = False

        # 5) 轻量适配器（Adapter）调整 img/text 全局特征
        self.use_adapters = False
        self.adapter_dim = 64
        self.adapter_scale = 0.2

        # 6) 质量残差门控（动态缩放 Δq）
        self.use_quality_gate = False

        # 7) 文本自适应调制（FiLM）
        self.use_film = False

        # 8) 一致性残差 MoE（多专家）
        self.use_moe_delta = False
        self.num_moe_experts = 2

        # 9) 文本token↔图像patch 跨注意力（更强交互）
        self.use_tp_cross_attn = False
        self.tp_cross_heads = 4
        self.tp_cross_layers = 1


class BaselineDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, proc: CLIPProcessor, tfm,
                 label_scale_q: float = 5.0, label_scale_c: float = 5.0):
        self.df, self.img_dir, self.proc, self.tfm = df, img_dir, proc, tfm
        self.label_scale_q = float(label_scale_q)
        self.label_scale_c = float(label_scale_c)
        # optional explanation/rationale text column
        self.exp_col = "explanation" if "explanation" in df.columns else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, r["name"])).convert("RGB")
        px = self.tfm(img)
        toks = self.proc(
            text=[r["prompt"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        # optional explanation tokens (for distillation)
        if self.exp_col is not None:
            exp_text = str(r[self.exp_col])
            exp_tok = self.proc(
                text=[exp_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77  # CLIP's max sequence length
            )
            exp_ids = exp_tok.input_ids[0]
            exp_mask = exp_tok.attention_mask[0]
        else:
            exp_ids = None
            exp_mask = None
        return (
            px,
            toks.input_ids[0],
            toks.attention_mask[0],
            torch.tensor(float(r["mos_quality"]) / self.label_scale_q, dtype=torch.float32),
            torch.tensor(float(r["mos_align"]) / self.label_scale_c, dtype=torch.float32),
            exp_ids,
            exp_mask,
        )


def collate_fn(batch):
    px = torch.stack([b[0] for b in batch])
    ids = nn.utils.rnn.pad_sequence([b[1] for b in batch], batch_first=True)
    mask = nn.utils.rnn.pad_sequence([b[2] for b in batch], batch_first=True)
    q = torch.stack([b[3] for b in batch]).unsqueeze(1)
    c = torch.stack([b[4] for b in batch]).unsqueeze(1)
    # explanation tokens are optional (may be None)
    has_exp = all(b[5] is not None for b in batch)
    if has_exp:
        exp_ids = nn.utils.rnn.pad_sequence([b[5] for b in batch], batch_first=True)
        exp_mask = nn.utils.rnn.pad_sequence([b[6] for b in batch], batch_first=True)
    else:
        exp_ids = None
        exp_mask = None
    return px, ids, mask, q, c, exp_ids, exp_mask


class ConsistencyRefinementModule(nn.Module):
    """Two-stage consistency refinement using lightweight Transformer.

    Stage 1: Coarse prediction from CLIP similarity
    Stage 2: Fine-grained refinement that learns residual corrections

    This module corrects CLIP's high/low bias and improves prediction smoothness.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, num_heads: int = 4):
        """
        Args:
            input_dim: Dimension of CLIP embeddings (e.g., 512 or 768)
            hidden_dim: Hidden dimension for transformer
            num_layers: Number of transformer encoder layers
            num_heads: Number of attention heads
        """
        super().__init__()

        # Project [img_emb, txt_emb, coarse_score] to hidden_dim
        # Input: [B, input_dim*2 + 1]
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim * 2 + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Lightweight Transformer for refinement
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Residual prediction head: outputs correction term
        # We use residual learning: refined = coarse + residual
        self.residual_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Bounded residual in [-1, 1]
        )

        # Learnable residual scale (starts small for stability)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor, coarse_score: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_emb: [B, dim] - normalized image embeddings
            txt_emb: [B, dim] - normalized text embeddings
            coarse_score: [B, 1] - coarse consistency score from stage 1

        Returns:
            refined_score: [B, 1] - refined consistency score (coarse + residual)
        """
        # Concatenate all inputs
        x = torch.cat([img_emb, txt_emb, coarse_score], dim=1)  # [B, dim*2+1]

        # Project to hidden dimension
        x = self.input_proj(x)  # [B, hidden_dim]

        # Add sequence dimension for transformer (treat as single token)
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]

        # Apply transformer refinement
        x = self.transformer(x)  # [B, 1, hidden_dim]

        # Remove sequence dimension
        x = x.squeeze(1)  # [B, hidden_dim]

        # Predict residual correction (bounded)
        residual = self.residual_head(x)  # [B, 1], range [-1, 1]
        residual = self.residual_scale * residual  # Scale down for stability

        # Residual learning: refined = coarse + residual
        refined_score = coarse_score + residual

        # Clip to valid range [0, 1]
        refined_score = torch.clamp(refined_score, 0.0, 1.0)

        return refined_score


class BaselineCLIPScore(nn.Module):
    """Use CLIP global image/text embeddings -> residual heads with optional modules."""

    def __init__(self, clip_model_name: str, freeze: bool = False,
                 use_refinement: bool = False, refinement_cfg: dict = None,
                 use_two_branch: bool = False, use_residual_learning: bool = True,
                 residual_scale_q: float = 0.2, residual_scale_c: float = 0.2,
                 partial_freeze: bool = False, freeze_layers: int = 8,
                 dropout: float = 0.1,
                 # 新增：可选架构模块
                 use_cross_attn_delta: bool = False,
                 cross_layers: int = 1,
                 cross_heads: int = 4,
                 cross_dim: int = 256,
                 dual_direction_cross: bool = False,
                 use_confidence_gate: bool = False,
                 use_token_quality: bool = False,
                 use_token_patch_cross: bool = False):
        super().__init__()
        self.clip = CLIPModel.from_pretrained(clip_model_name)
        
        # ===== 冻结策略 =====
        if freeze:
            # 完全冻结 CLIP
            for p in self.clip.parameters():
                p.requires_grad = False
        elif partial_freeze:
            # 部分冻结：只训练最后几层（渐进式微调）
            self._partial_freeze_clip(freeze_layers)
        
        dim = self.clip.config.projection_dim  # 512 for ViT-B/32
        
        # ===== 残差学习模式 =====
        self.use_residual_learning = use_residual_learning
        self.residual_scale_q = residual_scale_q
        self.residual_scale_c = residual_scale_c
        
        if use_residual_learning:
            # Quality: 基于 CLIP 图像特征的初始分数 + 残差微调
            # Base score 使用简单投影（可训练但小幅度）
            self.q_base_head = nn.Linear(dim, 1)
            # Delta predictor: 学习小的微调量
            self.q_delta_head = nn.Sequential(
                nn.Linear(dim, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Tanh()
            )
            # 质量门控（可选）
            self.quality_gate = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            # 传统模式：直接预测
            self.q_head = nn.Linear(dim, 1)

        self.use_refinement = use_refinement
        self.use_two_branch = use_two_branch

        # ===== Consistency 残差学习 =====
        if use_residual_learning:
            # Base: cosine similarity（CLIP 原始对齐分数）
            # 可选：添加可学习的 scale 和 bias（但保持小幅度）
            self.c_base_scale = nn.Parameter(torch.tensor(1.0))
            self.c_base_bias = nn.Parameter(torch.tensor(0.0))
            
            # Delta predictor: Fusion head 预测微调量
            # 输入：[img_emb, txt_emb, cos_sim]
            self.c_delta_head = nn.Sequential(
                nn.LayerNorm(dim * 2 + 1),
                nn.Linear(dim * 2 + 1, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Tanh()
            )

            # ===== MoE 残差专家（可选）=====
            self.use_moe_delta = False
            self.num_moe_experts = 2
            self.moe_experts = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(dim * 2 + 1),
                    nn.Linear(dim * 2 + 1, 128),
                    nn.GELU(),
                    nn.Linear(128, 1)
                ) for _ in range(self.num_moe_experts)
            ])
            self.moe_gate = nn.Sequential(
                nn.LayerNorm(dim * 2 + 1),
                nn.Linear(dim * 2 + 1, self.num_moe_experts)
            )

            # 扩展交互特征（|img-txt|, img*txt）
            # 输入维度 = [img_g, txt_g, sim] (2*dim+1) + [|img-txt|, img*txt] (2*dim) + [filip_sim] (1)
            self.ext_inter_proj = nn.Sequential(
                nn.LayerNorm(4 * dim + 2),
                nn.Linear(4 * dim + 2, 256),
                nn.GELU(),
                nn.Linear(256, 1),
                nn.Tanh()
            )

            # 为残差路径准备序列特征投影（用于计算 FILIP 相似度）
            v_dim = self.clip.config.vision_config.hidden_size
            t_dim = self.clip.config.text_config.hidden_size
            proj_dim = self.clip.config.projection_dim
            self.res_seq_proj_img = nn.Linear(v_dim, proj_dim, bias=False) if v_dim != proj_dim else nn.Identity()
            self.res_seq_proj_txt = nn.Linear(t_dim, proj_dim, bias=False) if t_dim != proj_dim else nn.Identity()

            # ===== 跨模态注意力一致性残差（模块化）=====
            self.use_cross_attn_delta = use_cross_attn_delta
            self.dual_direction_cross = dual_direction_cross
            if self.use_cross_attn_delta:
                self.cross_in = nn.Linear(dim, cross_dim, bias=False)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=cross_dim, nhead=cross_heads,
                    dim_feedforward=cross_dim * 2, dropout=0.1,
                    activation='gelu', batch_first=True
                )
                self.cross_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, cross_layers))
                if self.dual_direction_cross:
                    self.cross_delta = nn.Sequential(
                        nn.LayerNorm(2 * cross_dim),
                        nn.Linear(2 * cross_dim, cross_dim),
                        nn.GELU(),
                        nn.Linear(cross_dim, 1),
                        nn.Tanh()
                    )
                else:
                    self.cross_delta = nn.Sequential(
                        nn.LayerNorm(cross_dim),
                        nn.Linear(cross_dim, 1),
                        nn.Tanh()
                    )
            else:
                self.cross_in = None
                self.cross_encoder = None
                self.cross_delta = None

            # ===== 轻量 Token-Patch Cross（text token ↔ image patch）=====
            self.use_token_patch_cross = use_token_patch_cross
            v_dim = self.clip.config.vision_config.hidden_size
            t_dim = self.clip.config.text_config.hidden_size
            self.tp_img_proj = nn.Linear(v_dim, dim, bias=False) if v_dim != dim else nn.Identity()
            self.tp_txt_proj = nn.Linear(t_dim, dim, bias=False) if t_dim != dim else nn.Identity()
            self.tp_score_head = nn.Sequential(
                nn.LayerNorm(dim * 2 + 2),
                nn.Linear(dim * 2 + 2, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Tanh()
            )

            # 更强的 token↔patch Cross-Attention（可选）
            self.use_tp_cross_attn = False
            self.tp_cross_heads = 4
            self.tp_cross_layers = 1
            self.tp_cross_img_in = nn.Linear(v_dim, dim, bias=False) if v_dim != dim else nn.Identity()
            self.tp_cross_txt_in = nn.Linear(t_dim, dim, bias=False) if t_dim != dim else nn.Identity()
            ca_layer = nn.TransformerEncoderLayer(
                d_model=dim, nhead=self.tp_cross_heads,
                dim_feedforward=dim * 2, dropout=0.1,
                activation='gelu', batch_first=True
            )
            self.tp_cross_encoder = nn.TransformerEncoder(ca_layer, num_layers=self.tp_cross_layers)
            self.tp_cross_delta = nn.Sequential(
                nn.LayerNorm(dim * 2 + 2),
                nn.Linear(dim * 2 + 2, 128),
                nn.GELU(),
                nn.Linear(128, 1),
                nn.Tanh()
            )

            # ===== 置信度门控残差 =====
            self.use_confidence_gate = use_confidence_gate
            if self.use_confidence_gate:
                self.conf_gate = nn.Sequential(
                    nn.LayerNorm(dim * 2 + 1),
                    nn.Linear(dim * 2 + 1, 128),
                    nn.GELU(),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
            else:
                self.conf_gate = None

            # ===== 残差分支自适应混合器（Delta Mixer）=====
            self.use_delta_mixer = True
            self.delta_gate_mlp = nn.Sequential(
                nn.LayerNorm(dim * 2 + 1),
                nn.Linear(dim * 2 + 1, 128),
                nn.GELU()
            )
            self.delta_gate_heads = nn.ModuleDict({
                'mlp': nn.Linear(128, 1),
                'cross': nn.Linear(128, 1),
                'ext': nn.Linear(128, 1),
                'tp': nn.Linear(128, 1),
            })
        else:
            # 传统模式
            self.c_scale = nn.Parameter(torch.tensor(5.0))  # branch-A (global cos)
            self.c_bias = nn.Parameter(torch.tensor(0.0))

            # FILIP branch (token-patch interaction)
            self.filip_scale = nn.Parameter(torch.tensor(5.0))
            self.filip_bias = nn.Parameter(torch.tensor(0.0))

            # === sequence projection for FILIP (vision_hidden!=text_hidden) ===
            v_dim = self.clip.config.vision_config.hidden_size
            t_dim = self.clip.config.text_config.hidden_size
            proj_dim = self.clip.config.projection_dim  # 512
            self.seq_proj_img = nn.Linear(v_dim, proj_dim, bias=False) if v_dim != proj_dim else nn.Identity()
            self.seq_proj_txt = nn.Linear(t_dim, proj_dim, bias=False) if t_dim != proj_dim else nn.Identity()

            self.c_head = nn.Sequential(
                nn.LayerNorm(dim * 2 + 1),
                nn.Linear(dim * 2 + 1, 256),
                nn.GELU(),
                nn.Linear(256, 1)
            )

        # Rationale projection head: fuse [img, txt, cos] -> explanation embedding (dim)
        self.rationale_head = nn.Sequential(
            nn.LayerNorm(dim * 2 + 1),
            nn.Linear(dim * 2 + 1, dim)
        )

        # 轻量适配器（Adapter）以 residual 形式细调全局特征
        self.use_adapters = False
        self.img_adapter = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim)
        )
        self.txt_adapter = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, dim)
        )

        # 文本 FiLM 调制（可选）：用文本嵌入生成缩放与偏置，调制图像残差输入
        self.use_film = False
        self.film_gamma = nn.Linear(dim, dim)
        self.film_beta = nn.Linear(dim, dim)

        # Consistency Refinement Module (optional, two-stage)
        # 注意：残差学习模式下不再使用 refinement module，因为已经内置了残差机制
        if use_refinement and not use_residual_learning:
            refinement_cfg = refinement_cfg or {}
            self.refinement_module = ConsistencyRefinementModule(
                input_dim=dim,
                hidden_dim=refinement_cfg.get('hidden_dim', 256),
                num_layers=refinement_cfg.get('num_layers', 2),
                num_heads=refinement_cfg.get('num_heads', 4)
            )
        else:
            self.refinement_module = None

        # ===== token-level quality 分支（模块化）=====
        self.use_token_quality = use_token_quality
        v_dim = self.clip.config.vision_config.hidden_size
        if self.use_token_quality:
            self.q_seq_proj_img = nn.Linear(v_dim, dim, bias=False) if v_dim != dim else nn.Identity()
            self.q_patch_pool = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
            )
        else:
            self.q_seq_proj_img = None
            self.q_patch_pool = None
    
    def _partial_freeze_clip(self, freeze_layers: int):
        """部分冻结 CLIP：只训练最后几层编码器。
        
        Args:
            freeze_layers: 要冻结的层数（从头开始计数）
                          例如 freeze_layers=20 表示冻结前 20 层，只训练最后 4 层（ViT-L-14 有 24 层）
        """
        print(f"  [Partial Freeze] Freezing first {freeze_layers} layers of CLIP Vision and Text encoders")
        
        # 冻结 vision encoder 的前 N 层
        if hasattr(self.clip.vision_model, 'encoder'):
            encoder = self.clip.vision_model.encoder
            layers = getattr(encoder, 'layers', None) or getattr(encoder, 'layer', None)
            if layers is not None:
                for i, layer in enumerate(layers):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
        
        # 冻结 text encoder 的前 N 层
        if hasattr(self.clip.text_model, 'encoder'):
            encoder = self.clip.text_model.encoder
            layers = getattr(encoder, 'layers', None) or getattr(encoder, 'layer', None)
            if layers is not None:
                for i, layer in enumerate(layers):
                    if i < freeze_layers:
                        for param in layer.parameters():
                            param.requires_grad = False
        
        # 始终冻结 embeddings（位置编码等）
        if hasattr(self.clip.vision_model, 'embeddings'):
            for param in self.clip.vision_model.embeddings.parameters():
                param.requires_grad = False
        if hasattr(self.clip.text_model, 'embeddings'):
            for param in self.clip.text_model.embeddings.parameters():
                param.requires_grad = False

    def forward(self, pixel_values, ids, mask,
                use_cross_attn_delta: bool = False,
                cross_layers: int = 1,
                cross_heads: int = 4,
                cross_dim: int = 256,
                dual_direction_cross: bool = False,
                use_confidence_gate: bool = False,
                use_token_quality: bool = False,
                use_ext_interactions: bool = False,
                use_adapters: bool = False,
                adapter_scale: float = 0.2,
                use_quality_gate: bool = False,
                use_film: bool = False,
                use_moe_delta: bool = False):
        out = self.clip(pixel_values=pixel_values, input_ids=ids, attention_mask=mask, return_dict=True)
        img_g = F.normalize(out.image_embeds, dim=-1)  # [B,dim]
        txt_g = F.normalize(out.text_embeds, dim=-1)  # [B,dim]

        # 轻量适配器（可选）
        if use_adapters:
            img_g = F.normalize(img_g + adapter_scale * self.img_adapter(img_g), dim=-1)
            txt_g = F.normalize(txt_g + adapter_scale * self.txt_adapter(txt_g), dim=-1)
        
        # ===== Quality 预测（残差学习模式）=====
        if self.use_residual_learning:
            # Base score: 基于 CLIP 图像特征的简单投影
            q_base = torch.sigmoid(self.q_base_head(img_g))
            # Delta: 支持可选的 token-level 感知
            if self.use_token_quality and hasattr(out, 'vision_model_output') and out.vision_model_output is not None:
                img_seq = out.vision_model_output.last_hidden_state  # [B,Np+1,v_dim]
                # 丢弃 cls token（索引0），做简单的注意力权重近似：用 q_patch_pool 生成权重，再均值
                patches = img_seq[:, 1:, :]
                # 投到投影维度
                patches = self.q_seq_proj_img(patches) if self.q_seq_proj_img is not None else patches
                patches = F.normalize(patches, dim=-1)
                weights = self.q_patch_pool(patches) if self.q_patch_pool is not None else patches
                weights = weights.mean(dim=-1, keepdim=True)  # [B,Np,1]
                weights = torch.softmax(weights, dim=1)
                pooled = (patches * weights).sum(dim=1)  # [B,dim]
                if use_film:
                    gamma = self.film_gamma(txt_g)
                    beta = self.film_beta(txt_g)
                    pooled = pooled * (1 + torch.tanh(gamma)) + torch.tanh(beta)
                q_delta_raw = self.q_delta_head(pooled)
            else:
                pooled = img_g
                if use_film:
                    gamma = self.film_gamma(txt_g)
                    beta = self.film_beta(txt_g)
                    pooled = pooled * (1 + torch.tanh(gamma)) + torch.tanh(beta)
                q_delta_raw = self.q_delta_head(pooled)
            if use_quality_gate:
                q_gate = self.quality_gate(img_g)
            else:
                q_gate = 1.0
            q_delta = q_gate * q_delta_raw * self.residual_scale_q
            # 最终分数 = base + delta，并 clip 到 [0, 1]
            q = torch.clamp(q_base + q_delta, 0.0, 1.0)
        else:
            # 传统模式
            q = torch.sigmoid(self.q_head(img_g))

        # ===== Consistency 预测（残差学习模式）=====
        sim = (img_g * txt_g).sum(-1, keepdim=True)  # [B,1] cosine similarity
        
        if self.use_residual_learning:
            # ===== 纯残差学习模式（改进：可选跨模态注意力与置信度门控）=====
            # Base score: CLIP 的原始 cosine similarity（保留对齐空间）
            # 映射到 [0, 1] 区间：(sim + 1) / 2，因为 sim ∈ [-1, 1]
            c_base = (sim + 1.0) / 2.0  # [B,1]
            c_base = self.c_base_scale * c_base + self.c_base_bias
            c_base = torch.clamp(c_base, 0.0, 1.0)

            # Delta: 两种路径
            # A. 简单融合 MLP（默认）
            fused = torch.cat([img_g, txt_g, sim], dim=1)  # [B, dim*2+1]
            if use_moe_delta and len(self.moe_experts) > 0:
                logits = self.moe_gate(fused)
                weights = torch.softmax(logits, dim=-1)  # [B,E]
                expert_outs = torch.stack([exp(fused) for exp in self.moe_experts], dim=-1)  # [B,1,E]
                c_delta_mlp = torch.sum(expert_outs * weights.unsqueeze(1), dim=-1)  # [B,1]
                c_delta_mlp = torch.tanh(c_delta_mlp)  # [-1,1]
            else:
                c_delta_mlp = self.c_delta_head(fused)  # [-1,1]

            # B. 轻量跨模态注意力（可选）
            if (self.use_cross_attn_delta or use_cross_attn_delta) and self.cross_in is not None:
                seq = torch.stack([img_g, txt_g], dim=1)  # [B,2,dim]
                seq = self.cross_in(seq)  # [B,2,cross_dim]
                seq = self.cross_encoder(seq)  # [B,2,cross_dim]
                if (self.dual_direction_cross or dual_direction_cross):
                    cross_feat = seq.reshape(seq.size(0), -1)  # [B,2*cross_dim]
                    c_delta_cross = self.cross_delta(cross_feat)
                else:
                    cross_img = seq[:, 0, :]  # [B,cross_dim]
                    c_delta_cross = self.cross_delta(cross_img)
                c_delta_raw = 0.5 * (c_delta_mlp + c_delta_cross)
            else:
                c_delta_raw = c_delta_mlp

            # C. 扩展交互（可选）：加入 |img-txt|、img*txt 以及简化版 FILIP 相似度
            if use_ext_interactions and hasattr(out, 'vision_model_output') and hasattr(out, 'text_model_output'):
                img_seq = out.vision_model_output.last_hidden_state  # [B,Np+1,v_dim]
                txt_seq = out.text_model_output.last_hidden_state   # [B,Nt,t_dim]
                img_seq = self.res_seq_proj_img(img_seq)
                txt_seq = self.res_seq_proj_txt(txt_seq)
                img_seq = F.normalize(img_seq, dim=-1)
                txt_seq = F.normalize(txt_seq, dim=-1)
                # 取去除CLS的 patch/token
                img_patches = img_seq[:, 1:, :]  # [B,Np,dim]
                # 对每个文本token取与所有patch的最大相似，再做平均（text->img）
                t2i = torch.max(torch.einsum('bid,bjd->bij', txt_seq, img_patches), dim=2).values.mean(1, keepdim=True)
                # img->text（每个patch对所有token取最大相似再平均）
                i2t = torch.max(torch.einsum('bid,bjd->bij', img_patches, txt_seq), dim=2).values.mean(1, keepdim=True)
                filip_sim = 0.5 * (t2i + i2t)  # [B,1]

                inter_abs = torch.abs(img_g - txt_g)
                inter_mul = img_g * txt_g
                ext = torch.cat([img_g, txt_g, sim, inter_abs, inter_mul, filip_sim], dim=1)
                c_delta_ext = self.ext_inter_proj(ext)
                c_delta_raw = 0.5 * (c_delta_raw + c_delta_ext)

            # D. 轻量 Token-Patch Cross（可选）：基于 text token 与 image patch 局部对齐得分
            if self.use_token_patch_cross and hasattr(out, 'vision_model_output') and hasattr(out, 'text_model_output'):
                img_seq = out.vision_model_output.last_hidden_state  # [B,Np+1,v_dim]
                txt_seq = out.text_model_output.last_hidden_state   # [B,Nt,t_dim]
                img_patches = self.tp_img_proj(img_seq[:, 1:, :])  # [B,Np,dim]
                txt_tokens = self.tp_txt_proj(txt_seq)              # [B,Nt,dim]
                img_patches = F.normalize(img_patches, dim=-1)
                txt_tokens = F.normalize(txt_tokens, dim=-1)
                # 文本token -> 对应最相关的图像patch 相似度，取均值
                t2i_local = torch.max(torch.einsum('bid,bjd->bij', txt_tokens, img_patches), dim=2).values.mean(1, keepdim=True)  # [B,1]
                # 图像patch -> 文本token
                i2t_local = torch.max(torch.einsum('bid,bjd->bij', img_patches, txt_tokens), dim=2).values.mean(1, keepdim=True)  # [B,1]
                cross_local = 0.5 * (t2i_local + i2t_local)
                tp_feat = torch.cat([img_g, txt_g, sim, cross_local], dim=1)  # [B, 2*dim+2]
                c_delta_tp = self.tp_score_head(tp_feat)
                c_delta_raw = 0.5 * (c_delta_raw + c_delta_tp)

            # E. 更强 Token↔Patch Cross-Attention（可选）：拼接序列后通过小 Transformer 编码
            if self.use_tp_cross_attn and hasattr(out, 'vision_model_output') and hasattr(out, 'text_model_output'):
                img_seq = out.vision_model_output.last_hidden_state[:, 1:, :]  # [B,Np,v_dim]
                txt_seq = out.text_model_output.last_hidden_state              # [B,Nt,t_dim]
                img_seq = F.normalize(self.tp_cross_img_in(img_seq), dim=-1)
                txt_seq = F.normalize(self.tp_cross_txt_in(txt_seq), dim=-1)
                seq = torch.cat([txt_seq, img_seq], dim=1)  # [B,Nt+Np,dim]
                enc = self.tp_cross_encoder(seq)[:, 0, :]   # 取首 token 作为汇聚（粗略）
                tp_ca_feat = torch.cat([img_g, txt_g, sim, torch.tanh(enc.mean(dim=-1, keepdim=True))], dim=1)
                c_delta_tp2 = self.tp_cross_delta(tp_ca_feat)
                c_delta_raw = 0.5 * (c_delta_raw + c_delta_tp2)

            # 残差分支混合（learned gating over branches）
            if self.use_delta_mixer:
                gate_feat = self.delta_gate_mlp(fused)
                parts = {'mlp': c_delta_mlp}
                if (self.use_cross_attn_delta or use_cross_attn_delta) and 'c_delta_cross' in locals():
                    parts['cross'] = c_delta_cross
                if use_ext_interactions and 'c_delta_ext' in locals():
                    parts['ext'] = c_delta_ext
                if self.use_token_patch_cross and 'c_delta_tp' in locals():
                    parts['tp'] = c_delta_tp
                # 计算每个分支的 gate，然后归一化
                gates = []
                deltas = []
                for key, val in parts.items():
                    g = torch.sigmoid(self.delta_gate_heads[key](gate_feat))  # [B,1]
                    gates.append(g)
                    deltas.append(val)
                gate_sum = torch.clamp(sum(gates), min=1e-6)
                weights = [g / gate_sum for g in gates]
                c_delta_raw = sum(w * d for w, d in zip(weights, deltas))

            # 置信度门控（可选）：根据融合特征输出 gate ∈ [0,1] 对残差缩放
            if (self.use_confidence_gate or use_confidence_gate) and self.conf_gate is not None:
                gate = self.conf_gate(fused)  # [B,1]
            else:
                gate = 1.0

            c_delta = c_delta_raw * self.residual_scale_c * gate
            c = torch.clamp(c_base + c_delta, 0.0, 1.0)
            c_coarse = c_base
            
        else:
            # ===== 传统模式（多分支融合）=====
            # branch 1: scaled sigmoid of cosine (acts like a shortcut)
            cos_score = torch.sigmoid(self.c_scale * sim + self.c_bias)

            # ===== FILIP similarity =====
            # need patch/token sequences; request hidden states once
            img_seq = out.vision_model_output.last_hidden_state  # [B,Np+1,v_dim]
            txt_seq = out.text_model_output.last_hidden_state  # [B,Nt,t_dim]

            # project to shared dim
            img_seq = self.seq_proj_img(img_seq)
            txt_seq = self.seq_proj_txt(txt_seq)

            img_seq = F.normalize(img_seq, dim=-1)
            txt_seq = F.normalize(txt_seq, dim=-1)
            # compute per-pair filip similarity efficiently
            # text->img: for each token take max over patches
            t2i = torch.max(torch.einsum('bid,bjd->bij', txt_seq, img_seq), dim=2).values.mean(1, keepdim=True)
            # img->text: for each patch take max over tokens
            i2t = torch.max(torch.einsum('bid,bjd->bij', img_seq, txt_seq), dim=2).values.mean(1, keepdim=True)
            filip_sim = 0.5 * (t2i + i2t)  # [B,1]
            filip_score = torch.sigmoid(self.filip_scale * filip_sim + self.filip_bias)

            # branch 3: lightweight MLP on concatenated features+cos
            fused = torch.cat([img_g, txt_g, sim], dim=1)  # [B, 1025]
            mlp_score = torch.sigmoid(self.c_head(fused))

            # Stage 1: Coarse consistency prediction (average of three branches)
            if self.use_two_branch:
                c_coarse = 0.5 * (cos_score + mlp_score)
            else:
                c_coarse = mlp_score
            # Stage 2: Refinement (optional, learns residual corrections)
            if self.use_refinement and self.refinement_module is not None:
                c = self.refinement_module(img_g, txt_g, c_coarse)
            else:
                c = c_coarse

        return q, c, img_g, txt_g, c_coarse  # Return coarse/base score for debugging

    def compute_rationale_alignment_loss(self,
                                         img_g: torch.Tensor,
                                         txt_g: torch.Tensor,
                                         exp_ids: torch.Tensor,
                                         exp_mask: torch.Tensor) -> torch.Tensor:
        """Align fused [img, txt, cos] projection to explanation text embedding.

        Args:
            img_g: [B, dim] normalized image embeddings
            txt_g: [B, dim] normalized text embeddings (prompt)
            exp_ids/exp_mask: tokenized explanation batch
        Returns:
            Scalar MSE loss aligning projected fused features to explanation embedding.
        """
        if exp_ids is None or exp_mask is None:
            return torch.tensor(0.0, device=img_g.device)
        with torch.no_grad():
            # teacher target: explanation embedding from CLIP text tower (frozen by no_grad)
            exp_g = self.clip.get_text_features(input_ids=exp_ids, attention_mask=exp_mask)
            exp_g = F.normalize(exp_g, dim=-1)
        sim = (img_g * txt_g).sum(-1, keepdim=True)
        fused = torch.cat([img_g, txt_g, sim], dim=1)
        pred = self.rationale_head(fused)
        pred = F.normalize(pred, dim=-1)
        return F.mse_loss(pred, exp_g)


def train_epoch(model, dl, opt, sched, cfg):
    model.train()
    totals = [0.0, 0.0, 0.0, 0.0]  # total, lq, lc, le
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device.type == 'cuda'))

    opt.zero_grad(set_to_none=True)
    num_batches = len(dl)
    for step, batch in enumerate(dl):
        # unpack with optional explanation
        px, ids, mask, q_t, c_t, exp_ids, exp_mask = batch
        px = px.to(cfg.device, non_blocking=True)
        ids = ids.to(cfg.device, non_blocking=True)
        mask = mask.to(cfg.device, non_blocking=True)
        q_t = q_t.to(cfg.device, non_blocking=True)
        c_t = c_t.to(cfg.device, non_blocking=True)
        if exp_ids is not None:
            exp_ids = exp_ids.to(cfg.device, non_blocking=True)
            exp_mask = exp_mask.to(cfg.device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(cfg.use_amp and cfg.device.type == 'cuda')):
            q_p, c_p, img_g, txt_g, c_coarse = model(px, ids, mask)
            lq = F.mse_loss(q_p, q_t)

            # ===== Residual Learning for Refinement Module =====
            if cfg.use_refinement and model.use_refinement and cfg.strict_residual:
                # Strict residual learning: supervise the residual directly
                target_residual = c_t - c_coarse.detach()
                predicted_residual = c_p - c_coarse.detach()
                lc = F.mse_loss(predicted_residual, target_residual)
            else:
                # Standard learning: supervise the final score
                lc = F.mse_loss(c_p, c_t)

            le = torch.tensor(0.0, device=cfg.device)
            if cfg.use_explanations and exp_ids is not None:
                le = model.compute_rationale_alignment_loss(img_g, txt_g, exp_ids, exp_mask)

            loss = cfg.w_q * lq + cfg.w_c * lc + (cfg.w_exp * le if cfg.use_explanations and exp_ids is not None else 0.0)

        # normalize by grad_accum_steps to keep loss scale
        loss_to_backprop = loss / max(1, cfg.grad_accum_steps)

        if scaler.is_enabled():
            scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        do_step = ((step + 1) % cfg.grad_accum_steps == 0) or ((step + 1) == num_batches)
        if do_step:
            # 梯度裁剪（防止梯度爆炸）
            if cfg.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            if sched is not None:
                # scheduler 按更新步数推进（per optimizer step）
                sched.step()

        # 统计（以 batch 粒度汇总）
        totals[0] += float(loss.item())
        totals[1] += float(lq.item())
        totals[2] += float(lc.item())
        totals[3] += float(le.item()) if torch.is_tensor(le) else 0.0

    n = len(dl)
    return [t / max(1, n) for t in totals]


def evaluate(model, dl, cfg, print_examples=False, num_examples=5):
    model.eval();
    preds_q, tgts_q, preds_c, tgts_c, preds_c_coarse = [], [], [], [], []

    with torch.no_grad():
        for px, ids, mask, q_t, c_t, _, _ in dl:
            px, ids, mask = px.to(cfg.device), ids.to(cfg.device), mask.to(cfg.device)
            q_p, c_p, _, _, c_coarse = model(px, ids, mask)
            preds_q.extend(q_p.cpu().numpy().flatten());
            tgts_q.extend(q_t.numpy().flatten())
            preds_c.extend(c_p.cpu().numpy().flatten());
            tgts_c.extend(c_t.numpy().flatten())
            preds_c_coarse.extend(c_coarse.cpu().numpy().flatten())

    s_q = spearmanr(tgts_q, preds_q).correlation;
    s_c = spearmanr(tgts_c, preds_c).correlation
    p_q = pearsonr(tgts_q, preds_q)[0];
    p_c = pearsonr(tgts_c, preds_c)[0]

    # Print example predictions to show refinement effect
    if print_examples and cfg.use_refinement:
        print(f"\n  {'=' * 70}")
        print(f"  Example Predictions (Coarse vs Refined vs Ground Truth):")
        print(f"  {'=' * 70}")
        print(f"  {'ID':<5} {'Coarse':>10} {'Refined':>10} {'GT':>10} {'Δ(Ref-Coa)':>12} {'Error':>10}")
        print(f"  {'-' * 70}")

        # Show a few examples
        indices = list(range(min(num_examples, len(tgts_c))))
        for i in indices:
            coarse_val = preds_c_coarse[i]
            refined_val = preds_c[i]
            gt_val = tgts_c[i]
            delta = refined_val - coarse_val
            error = abs(refined_val - gt_val)

            print(f"  {i + 1:<5} {coarse_val:>10.4f} {refined_val:>10.4f} {gt_val:>10.4f} "
                  f"{delta:>+12.4f} {error:>10.4f}")

        # Statistics
        deltas = [preds_c[i] - preds_c_coarse[i] for i in range(len(preds_c))]
        avg_delta = sum(deltas) / len(deltas)
        max_delta = max(deltas)
        min_delta = min(deltas)

        print(f"  {'-' * 70}")
        print(f"  Refinement Stats: Avg Δ={avg_delta:+.4f}, Min Δ={min_delta:+.4f}, Max Δ={max_delta:+.4f}")
        print(f"  {'=' * 70}\n")

    return s_q, p_q, s_c, p_c


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Try to keep training reasonably deterministic without forcing slow algos
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def build_scheduler(optimizer, cfg, total_update_steps: int):
    scheduler_name = (cfg.scheduler or "cosine").lower()
    warmup_steps = int(cfg.warmup_ratio * total_update_steps)
    if scheduler_name == "cosine":
        sched = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)
        info = f"cosine (warmup_steps={warmup_steps}, total_steps={total_update_steps})"
        step_on = "step"
    elif scheduler_name == "linear":
        sched = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_update_steps)
        info = f"linear (warmup_steps={warmup_steps}, total_steps={total_update_steps})"
        step_on = "step"
    elif scheduler_name == "constant":
        sched = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        info = f"constant (warmup_steps={warmup_steps})"
        step_on = "step"
    elif scheduler_name == "step":
        # Note: StepLR 不基于步数总量，此处按每 epoch step 的方式使用
        sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.step_lr_step_size), gamma=cfg.step_lr_gamma)
        info = f"step (step_size={cfg.step_lr_step_size} epoch, gamma={cfg.step_lr_gamma})"
        step_on = "epoch"
    else:
        sched = None
        info = "None"
        step_on = "none"
    return sched, info, step_on


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(path: str, model, optimizer, scheduler, epoch: int, best_sc: float):
    to_save = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if (scheduler is not None and hasattr(scheduler, 'state_dict')) else None,
        "epoch": epoch,
        "best_sc": best_sc,
        "timestamp": datetime.now().isoformat(timespec='seconds')
    }
    torch.save(to_save, path)


def load_checkpoint(path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            pass
    start_epoch = int(ckpt.get("epoch", 0))
    best_sc = float(ckpt.get("best_sc", -1.0))
    return start_epoch, best_sc


def main():
    cfg = TrainingConfig()
    parser = argparse.ArgumentParser("Baseline CLIP eval")
    parser.add_argument('--data_csv_path');
    parser.add_argument('--image_base_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--epochs', type=int);
    parser.add_argument('--batch_size', type=int);
    parser.add_argument('--lr', type=float)
    parser.add_argument('--w_q', type=float);
    parser.add_argument('--w_c', type=float);
    parser.add_argument('--freeze_clip', action='store_true')
    parser.add_argument('--use_explanations', action='store_true')
    parser.add_argument('--use_two_branch', action='store_true')
    parser.add_argument('--w_exp', type=float)
    parser.add_argument('--explanation_column')
    parser.add_argument('--use_refinement', action='store_true', help='Enable two-stage consistency refinement')
    parser.add_argument('--refinement_layers', type=int, help='Number of transformer layers in refinement')
    parser.add_argument('--refinement_heads', type=int, help='Number of attention heads in refinement')
    parser.add_argument('--refinement_dim', type=int, help='Hidden dimension for refinement module')
    parser.add_argument('--no_strict_residual', action='store_true',
                        help='Disable strict residual learning (default: enabled)')
    
    # ===== 新增：残差学习和部分冻结参数 =====
    parser.add_argument('--no_residual_learning', action='store_true',
                        help='禁用残差学习模式（默认启用）')
    parser.add_argument('--residual_scale_q', type=float,
                        help='Quality 残差缩放因子（默认 0.2）')
    parser.add_argument('--residual_scale_c', type=float,
                        help='Consistency 残差缩放因子（默认 0.2）')
    parser.add_argument('--partial_freeze', action='store_true',
                        help='启用部分冻结 CLIP（只训练最后几层）')
    parser.add_argument('--freeze_layers', type=int,
                        help='冻结前 N 层编码器（默认 8）')
    
    # ===== 新增：训练优化参数 =====
    parser.add_argument('--weight_decay', type=float,
                        help='L2 正则化权重（默认 1e-4）')
    parser.add_argument('--warmup_ratio', type=float,
                        help='Warmup steps 占总步数比例（默认 0.05）')
    parser.add_argument('--max_grad_norm', type=float,
                        help='梯度裁剪阈值，0 表示不裁剪（默认 1.0）')
    parser.add_argument('--dropout', type=float,
                        help='Dropout 比例（默认 0.1）')
    parser.add_argument('--grad_accum_steps', type=int, help='梯度累积步数（默认 1）')
    parser.add_argument('--no_amp', action='store_true', help='禁用 AMP 混合精度（默认启用，如果有 CUDA）')
    parser.add_argument('--seed', type=int, help='随机种子（默认 42）')
    parser.add_argument('--num_workers', type=int, help='DataLoader 工作进程数（默认 4）')
    parser.add_argument('--pin_memory', action='store_true', help='DataLoader 使用 pin_memory')
    parser.add_argument('--no_pin_memory', action='store_true', help='禁用 pin_memory')
    parser.add_argument('--scheduler', type=str, choices=['cosine', 'linear', 'constant', 'step'],
                        help='学习率调度器类型（默认 cosine）')
    parser.add_argument('--step_lr_step_size', type=int, help='StepLR: 衰减步长（单位：epoch）')
    parser.add_argument('--step_lr_gamma', type=float, help='StepLR: 衰减因子（默认 0.1）')
    parser.add_argument('--resume_from', type=str, help='从检查点恢复训练（.pt 文件路径）')
    parser.add_argument('--log_csv', type=str, help='训练日志 CSV 文件名（默认 training_log.csv）')
    parser.add_argument('--label_scale_q', type=float, help='Quality 标签缩放（将标签除以该值，默认自动）')
    parser.add_argument('--label_scale_c', type=float, help='Consistency 标签缩放（将标签除以该值，默认自动）')
    parser.add_argument('--early_stopping_patience', type=int, help='早停耐心值（0 表示不启用）')
    parser.add_argument('--early_stopping_min_delta', type=float, help='早停最小提升阈值（默认 1e-4）')

    # ===== 新增：架构模块 CLI 开关 =====
    parser.add_argument('--use_cross_attn_delta', action='store_true', help='启用跨模态注意力一致性残差')
    parser.add_argument('--cross_layers', type=int, help='跨模态注意力层数（默认1）')
    parser.add_argument('--cross_heads', type=int, help='跨模态注意力头数（默认4）')
    parser.add_argument('--cross_dim', type=int, help='跨模态注意力隐藏维度（默认256）')
    parser.add_argument('--dual_direction_cross', action='store_true', help='跨模态注意力使用双向聚合')
    parser.add_argument('--use_confidence_gate', action='store_true', help='启用置信度门控残差')
    parser.add_argument('--use_token_quality', action='store_true', help='启用基于patch的质量残差输入')
    parser.add_argument('--use_ext_interactions', action='store_true', help='一致性残差使用扩展交互特征')
    parser.add_argument('--use_adapters', action='store_true', help='启用轻量适配器细调全局表征')
    parser.add_argument('--adapter_scale', type=float, help='适配器残差缩放（默认0.2）')
    parser.add_argument('--use_quality_gate', action='store_true', help='启用质量残差门控')
    parser.add_argument('--use_token_patch_cross', action='store_true', help='启用文本token与图像patch的轻量交互')
    
    args = parser.parse_args()
    if args.data_csv_path: cfg.data_csv_path = args.data_csv_path
    if args.image_base_dir: cfg.image_base_dir = args.image_base_dir
    if args.output_dir: cfg.output_dir = args.output_dir
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr: cfg.lr = args.lr
    if args.w_q: cfg.w_q = args.w_q
    if args.w_c: cfg.w_c = args.w_c
    if args.freeze_clip: cfg.freeze_clip = True
    if args.use_explanations: cfg.use_explanations = True
    if args.w_exp is not None: cfg.w_exp = args.w_exp
    if args.explanation_column: cfg.explanation_column = args.explanation_column
    if args.use_refinement: cfg.use_refinement = True
    if args.refinement_layers: cfg.refinement_layers = args.refinement_layers
    if args.refinement_heads: cfg.refinement_heads = args.refinement_heads
    if args.refinement_dim: cfg.refinement_dim = args.refinement_dim
    if args.no_strict_residual: cfg.strict_residual = False
    
    # 残差学习和部分冻结参数
    if args.no_residual_learning: cfg.use_residual_learning = False
    if args.residual_scale_q is not None: cfg.residual_scale_q = args.residual_scale_q
    if args.residual_scale_c is not None: cfg.residual_scale_c = args.residual_scale_c
    if args.partial_freeze: cfg.partial_freeze = True
    if args.freeze_layers: cfg.freeze_layers = args.freeze_layers
    
    # 训练优化参数
    if args.weight_decay is not None: cfg.weight_decay = args.weight_decay
    if args.warmup_ratio is not None: cfg.warmup_ratio = args.warmup_ratio
    if args.max_grad_norm is not None: cfg.max_grad_norm = args.max_grad_norm
    if args.dropout is not None: cfg.dropout = args.dropout
    if args.grad_accum_steps is not None and args.grad_accum_steps > 0: cfg.grad_accum_steps = args.grad_accum_steps
    if args.no_amp: cfg.use_amp = False
    if args.seed is not None: cfg.seed = args.seed
    if args.num_workers is not None: cfg.num_workers = max(0, args.num_workers)
    if args.pin_memory: cfg.pin_memory = True
    if args.no_pin_memory: cfg.pin_memory = False
    if args.scheduler is not None: cfg.scheduler = args.scheduler
    if args.step_lr_step_size is not None: cfg.step_lr_step_size = max(1, args.step_lr_step_size)
    if args.step_lr_gamma is not None: cfg.step_lr_gamma = args.step_lr_gamma
    if args.resume_from: cfg.resume_from = args.resume_from
    if args.log_csv: cfg.log_csv = args.log_csv
    if args.label_scale_q is not None: cfg.label_scale_q = float(args.label_scale_q)
    if args.label_scale_c is not None: cfg.label_scale_c = float(args.label_scale_c)
    if args.early_stopping_patience is not None: cfg.early_stopping_patience = max(0, int(args.early_stopping_patience))
    if args.early_stopping_min_delta is not None: cfg.early_stopping_min_delta = float(args.early_stopping_min_delta)

    # 架构模块参数
    if args.use_cross_attn_delta: cfg.use_cross_attn_delta = True
    if args.cross_layers is not None: cfg.cross_layers = max(1, args.cross_layers)
    if args.cross_heads is not None: cfg.cross_heads = max(1, args.cross_heads)
    if args.cross_dim is not None: cfg.cross_dim = max(32, args.cross_dim)
    if args.dual_direction_cross: cfg.dual_direction_cross = True
    if args.use_confidence_gate: cfg.use_confidence_gate = True
    if args.use_token_quality: cfg.use_token_quality = True
    if args.use_ext_interactions: cfg.use_ext_interactions = True
    if args.use_adapters: cfg.use_adapters = True
    if args.adapter_scale is not None: cfg.adapter_scale = float(args.adapter_scale)
    if args.use_quality_gate: cfg.use_quality_gate = True
    if args.use_token_patch_cross: cfg.use_token_patch_cross = True

    # Prepare output directory
    ensure_dir(cfg.output_dir)

    # Set seeds and precision behavior
    set_seed(cfg.seed)
    if hasattr(torch, 'set_float32_matmul_precision'):
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Load data
    df = pd.read_csv(cfg.data_csv_path)
    # Auto-detect label scale if not provided
    if cfg.label_scale_q is None:
        # If max quality >1.5 we assume 1-5 scale
        try:
            max_q = float(pd.to_numeric(df["mos_quality"], errors='coerce').max())
            cfg.label_scale_q = 5.0 if max_q > 1.5 else 1.0
        except Exception:
            cfg.label_scale_q = 5.0
    if cfg.label_scale_c is None:
        try:
            max_c = float(pd.to_numeric(df["mos_align"], errors='coerce').max())
            cfg.label_scale_c = 5.0 if max_c > 1.5 else 1.0
        except Exception:
            cfg.label_scale_c = 5.0
    train_df, val_df = train_test_split(df, test_size=cfg.test_size, random_state=42)
    proc = CLIPProcessor.from_pretrained(cfg.clip_model_name)
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(cfg.image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_processor.image_mean, proc.image_processor.image_std)
    ])
    tf_val = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(proc.image_processor.image_mean, proc.image_processor.image_std)
    ])
    # If a custom explanation column is provided, ensure it exists (optional)
    if cfg.explanation_column != "explanation" and cfg.explanation_column in train_df.columns:
        train_df = train_df.rename(columns={cfg.explanation_column: "explanation"})
        val_df = val_df.rename(columns={cfg.explanation_column: "explanation"})
    train_ds = BaselineDataset(train_df, cfg.image_base_dir, proc, tf_train,
                               label_scale_q=cfg.label_scale_q, label_scale_c=cfg.label_scale_c)
    val_ds = BaselineDataset(val_df, cfg.image_base_dir, proc, tf_val,
                             label_scale_q=cfg.label_scale_q, label_scale_c=cfg.label_scale_c)
    # DataLoader settings
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        drop_last=False,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        drop_last=False,
    )

    # Build refinement config
    refinement_cfg = {
        'hidden_dim': cfg.refinement_dim,
        'num_layers': cfg.refinement_layers,
        'num_heads': cfg.refinement_heads
    } if cfg.use_refinement else None

    model = BaselineCLIPScore(
        cfg.clip_model_name,
        freeze=cfg.freeze_clip,
        use_refinement=cfg.use_refinement,
        refinement_cfg=refinement_cfg,
        use_two_branch=cfg.use_two_branch,
        use_residual_learning=cfg.use_residual_learning,
        residual_scale_q=cfg.residual_scale_q,
        residual_scale_c=cfg.residual_scale_c,
        partial_freeze=cfg.partial_freeze,
        freeze_layers=cfg.freeze_layers,
        dropout=cfg.dropout,
        use_cross_attn_delta=cfg.use_cross_attn_delta,
        cross_layers=cfg.cross_layers,
        cross_heads=cfg.cross_heads,
        cross_dim=cfg.cross_dim,
        dual_direction_cross=cfg.dual_direction_cross,
        use_confidence_gate=cfg.use_confidence_gate,
        use_token_quality=cfg.use_token_quality,
        use_token_patch_cross=cfg.use_token_patch_cross
    ).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 学习率调度器（支持多种策略），步数需考虑梯度累积
    updates_per_epoch = math.ceil(len(train_dl) / max(1, cfg.grad_accum_steps))
    total_update_steps = max(1, updates_per_epoch * cfg.epochs)
    sched, sched_info, sched_step_on = build_scheduler(opt, cfg, total_update_steps)

    # Print config
    print(f"\n{'=' * 70}")
    print(f"Training Configuration:")
    print(f"  - Epochs: {cfg.epochs}, Batch Size: {cfg.batch_size}, LR: {cfg.lr}")
    print(f"  - Optimizer: AdamW (weight_decay={cfg.weight_decay})")
    print(f"  - Scheduler: {sched_info}")
    print(f"  - Updates/epoch: {updates_per_epoch}, Total updates: {total_update_steps}")
    print(f"  - Regularization: dropout={cfg.dropout}, max_grad_norm={cfg.max_grad_norm}")
    print(f"  - AMP: {cfg.use_amp and cfg.device.type == 'cuda'}, GradAccum: {cfg.grad_accum_steps}")
    print(f"  - DataLoader: workers={cfg.num_workers}, pin_memory={cfg.pin_memory}")
    print(f"  - Output Dir: {cfg.output_dir}")
    print(f"  - Loss Weights: w_q={cfg.w_q}, w_c={cfg.w_c}")
    print(f"  - Label scale: q={cfg.label_scale_q}, c={cfg.label_scale_c}")
    print(f"  - Use Explanations: {cfg.use_explanations}" + (f" (w_exp={cfg.w_exp})" if cfg.use_explanations else ""))
    print(f"\n  🔧 CLIP 冻结策略:")
    if cfg.freeze_clip:
        print(f"     ❄️  完全冻结 CLIP（Linear Probing）")
    elif cfg.partial_freeze:
        print(f"     🧊 部分冻结：前 {cfg.freeze_layers} 层冻结，其余可训练")
    else:
        print(f"     🔥 CLIP 完全可训练（端到端微调）")
    
    print(f"\n  🎯 预测架构:")
    if cfg.use_residual_learning:
        print(f"     ✅ 残差学习模式（保留 CLIP 对齐空间）")
        print(f"        - Quality:  q = q_base + Δq × {cfg.residual_scale_q}")
        print(f"        - Consistency: c = cos(img,txt) + Δc × {cfg.residual_scale_c}")
        print(f"        - 原理: 只学习微调量，防止破坏 CLIP 原始空间")
    else:
        print(f"     ⚠️  传统模式（直接预测，可能偏离 CLIP 空间）")
        if cfg.use_refinement:
            residual_mode = "Strict Residual" if cfg.strict_residual else "Standard"
            print(f"        - Use Refinement: True ({cfg.refinement_layers}L, {cfg.refinement_heads}H, dim={cfg.refinement_dim}, {residual_mode})")

    print(f"{'=' * 70}\n")

    # Resume support
    start_epoch = 0
    best_sc = -1.0
    if cfg.resume_from is not None and os.path.isfile(cfg.resume_from):
        try:
            start_epoch, best_sc = load_checkpoint(cfg.resume_from, model, opt, sched, cfg.device)
            print(f"[Resume] Loaded checkpoint from {cfg.resume_from} (start_epoch={start_epoch}, best_sc={best_sc:.4f})")
        except Exception as e:
            print(f"[Resume] Failed to load checkpoint: {e}")

    # Prepare CSV logging
    log_path = os.path.join(cfg.output_dir, cfg.log_csv)
    write_header = not os.path.exists(log_path)

    patience_counter = 0
    for ep in range(start_epoch, cfg.epochs):
        train_loss, lq, lc, le = train_epoch(model, train_dl, opt, sched if (sched is not None and sched_step_on == 'step') else None, cfg)

        # If scheduler is StepLR stepped per-epoch
        if sched is not None and sched_step_on == 'epoch':
            sched.step()

        # Print examples every 5 epochs or at the last epoch
        print_examples = cfg.use_refinement and ((ep + 1) % 5 == 0 or (ep + 1) == cfg.epochs)
        s_q, p_q, s_c, p_c = evaluate(model, val_dl, cfg, print_examples=print_examples, num_examples=5)

        cur_lr = opt.param_groups[0]['lr']
        if cfg.use_explanations:
            print(
                f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f},E{le:.4f})  Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f}  LR={cur_lr:.2e}")
        else:
            print(
                f"Ep{ep + 1} TrainLoss={train_loss:.4f}(Q{lq:.4f},C{lc:.4f})  Val SROCC_Q={s_q:.4f},SROCC_C={s_c:.4f}  LR={cur_lr:.2e}")

        # Save last checkpoint every epoch
        last_path = os.path.join(cfg.output_dir, "baseline_last.pt")
        save_checkpoint(last_path, model, opt, sched, ep + 1, best_sc)

        # Save best by SROCC_C
        improved = (s_c - best_sc) > cfg.early_stopping_min_delta
        if improved:
            best_sc = s_c
            patience_counter = 0
            # 根据配置生成模型文件名
            if cfg.use_residual_learning:
                if cfg.partial_freeze:
                    save_name = f"baseline_residual_partial_freeze_{cfg.freeze_layers}L_best.pt"
                else:
                    save_name = "baseline_residual_best.pt"
            elif cfg.use_refinement:
                save_name = "baseline_refinement_best.pt"
            else:
                save_name = "baseline_best.pt"
            best_path = os.path.join(cfg.output_dir, save_name)
            torch.save(model.state_dict(), best_path)
            # Also save a full checkpoint for resume
            best_ckpt_path = os.path.join(cfg.output_dir, "baseline_best.pt")
            save_checkpoint(best_ckpt_path, model, opt, sched, ep + 1, best_sc)
            print(f"[Checkpoint] New best SROCC_C={s_c:.4f} -> {best_path}")
        else:
            patience_counter += 1

        # Append CSV log
        with open(log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["epoch", "train_loss", "lq", "lc", "le", "s_q", "p_q", "s_c", "p_c", "best_s_c", "lr"]) 
                write_header = False
            writer.writerow([ep + 1, f"{train_loss:.6f}", f"{lq:.6f}", f"{lc:.6f}", f"{le:.6f}", f"{s_q:.6f}", f"{p_q:.6f}", f"{s_c:.6f}", f"{p_c:.6f}", f"{best_sc:.6f}", f"{cur_lr:.8f}"])

        # Early stopping
        if cfg.early_stopping_patience and patience_counter >= cfg.early_stopping_patience:
            print(f"[EarlyStopping] No improvement for {patience_counter} epochs (patience={cfg.early_stopping_patience}). Stopping.")
            break

    # Final evaluation with examples
    if cfg.use_refinement:
        print("\n" + "=" * 70)
        print("FINAL EVALUATION WITH EXAMPLES")
        print("=" * 70)
        s_q, p_q, s_c, p_c = evaluate(model, val_dl, cfg, print_examples=True, num_examples=10)

    print("\nTraining Complete!")
    print(f"Best SROCC_C: {best_sc:.4f}")


if __name__ == "__main__":
    main()
