#!/usr/bin/env python3
"""
TiTok Fine-tuning Script for ACDC MRI Dataset

对TiTok模型在ACDC心脏MRI数据集上进行微调
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from PIL import Image
import argparse
import yaml
from datetime import datetime
import logging
from tqdm import tqdm
import warnings
from typing import Tuple, Dict, Any, Optional
from omegaconf import OmegaConf

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入必要的模块
from modeling.titok import TiTok
from modeling.maskgit import ImageBert
from med_mri.acdc_dataset import create_data_loaders, ACDCMRIDataset
import demo_util

warnings.filterwarnings("ignore")


class TiTokMRIWrapper(nn.Module):
    """
    TiTok模型包装器，用于MRI微调
    """

    def __init__(self, tokenizer_path: str, generator_path: str = None, device: str = 'cuda'):
        """
        初始化TiTok MRI包装器

        Args:
            tokenizer_path: tokenizer checkpoint路径
            generator_path: generator checkpoint路径 (可选)
            device: 计算设备
        """
        super().__init__()

        self.device = device
        self.tokenizer_path = tokenizer_path
        self.generator_path = generator_path

        # 加载tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        self.tokenizer.eval()

        # 加载generator (如果提供)
        if generator_path and os.path.exists(generator_path):
            self.generator = self._load_generator(generator_path)
            self.generator.eval()
        else:
            self.generator = None

        print(f"✅ TiTok模型加载完成 - Tokenizer: {tokenizer_path}")
        if self.generator:
            print(f"✅ Generator: {generator_path}")

    def _load_tokenizer(self, checkpoint_path: str) -> TiTok:
        """加载TiTok tokenizer"""
        model_name = self._get_model_name_from_path(checkpoint_path)
        try:
            tokenizer = TiTok.from_pretrained(model_name)
            print(f"✅ 从HuggingFace加载tokenizer: {model_name}")
        except Exception as e:
            print(f"⚠️ HuggingFace加载失败，尝试本地加载: {e}")
            tokenizer = TiTok.from_pretrained(checkpoint_path)

        return tokenizer

    def _load_generator(self, checkpoint_path: str) -> ImageBert:
        """加载generator"""
        model_name = self._get_model_name_from_path(checkpoint_path)
        try:
            generator = ImageBert.from_pretrained(model_name)
            print(f"✅ 从HuggingFace加载generator: {model_name}")
        except Exception as e:
            print(f"⚠️ Generator加载失败: {e}")
            return None

        return generator

    def _get_model_name_from_path(self, path: str) -> str:
        """从路径提取模型名称"""
        if 'b64' in path.lower():
            return 'yucornetto/tokenizer_titok_b64_imagenet'
        elif 'b128' in path.lower():
            return 'yucornetto/tokenizer_titok_b128_imagenet'
        return 'yucornetto/tokenizer_titok_b64_imagenet'

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播: 编码 -> 解码"""
        # 编码为token (returns (tokens, info_dict))
        with torch.no_grad():
            encode_result = self.tokenizer.encode(x)
            if isinstance(encode_result, tuple):
                tokens, _ = encode_result  # tokens shape: [B, C, 1, 64]
            else:
                tokens = encode_result

        # 解码重建图像
        with torch.no_grad():
            if self.generator is not None:
                reconstructed = self.generator.decode(tokens)
            else:
                # 使用tokenizer的解码器
                decode_result = self.tokenizer.decode(tokens)
                # Handle both tensor and tuple returns
                if isinstance(decode_result, tuple):
                    reconstructed = decode_result[0]
                else:
                    reconstructed = decode_result

        return reconstructed, tokens

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """仅编码"""
        with torch.no_grad():
            tokens = self.tokenizer.encode(x)
        return tokens

    def to(self, device):
        """移动到设备"""
        self.device = device
        self.tokenizer = self.tokenizer.to(device)
        if self.generator is not None:
            self.generator = self.generator.to(device)
        return super().to(device)


class TiTokMRIEvaluator:
    """评估器"""

    def __init__(self):
        pass

    def compute_metrics(self, images, reconstructed, tokens):
        """计算重建指标"""
        # MSE
        mse = torch.mean((images - reconstructed) ** 2).item()

        # PSNR
        max_val = 1.0
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else 100.0

        # 压缩率
        original_pixels = images.shape[-2] * images.shape[-1] * images.shape[-3]
        n_tokens = tokens.shape[-1]
        compression_ratio = original_pixels / n_tokens

        return {
            'mse': mse,
            'psnr': psnr,
            'compression_ratio': compression_ratio
        }


def train_one_epoch(
    model: TiTokMRIWrapper,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]")

    for batch in pbar:
        images = batch['image'].to(device)

        # 前向传播
        reconstructed, tokens = model(images)

        # 重建损失
        loss = nn.MSELoss()(reconstructed, images)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({'loss': total_loss / n_batches})

    return {'train_loss': total_loss / n_batches}


def validate(
    model: TiTokMRIWrapper,
    val_loader: DataLoader,
    device: str,
    epoch: int,
    evaluator: TiTokMRIEvaluator,
) -> Dict[str, float]:
    """验证"""
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validating")
        for batch in pbar:
            images = batch['image'].to(device)
            reconstructed, tokens = model(images)

            metrics = evaluator.compute_metrics(images, reconstructed, tokens)
            total_mse += metrics['mse']
            total_psnr += metrics['psnr']
            n_batches += 1

            pbar.set_postfix({'mse': total_mse / n_batches, 'psnr': total_psnr / n_batches})

    return {
        'val_mse': total_mse / n_batches,
        'val_psnr': total_psnr / n_batches
    }


def main(args):
    """主训练函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 设备
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    logger.info(f"使用设备: {device}")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 创建数据加载器
    logger.info("创建数据加载器...")
    data_loaders = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4 if device == 'cuda' else 0,
        image_size=(256, 256),
        augment=True
    )

    # 创建模型
    logger.info("创建TiTok模型...")
    model = TiTokMRIWrapper(
        tokenizer_path=args.tokenizer_path,
        generator_path=args.generator_path,
        device=device
    ).to(device)

    # 冻结generator，只训练tokenizer
    if model.generator is not None:
        for param in model.generator.parameters():
            param.requires_grad = False

    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # 评估器
    evaluator = TiTokMRIEvaluator()

    # 训练循环
    best_val_loss = float('inf')
    logger.info("开始训练...")

    for epoch in range(args.num_epochs):
        # 训练
        train_metrics = train_one_epoch(
            model,
            data_loaders['train'],
            optimizer,
            device,
            epoch,
            args.num_epochs
        )

        # 验证
        val_metrics = validate(
            model,
            data_loaders['val'],
            device,
            epoch,
            evaluator
        )

        scheduler.step()

        # 日志
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_metrics['train_loss']:.6f} - "
            f"Val MSE: {val_metrics['val_mse']:.6f} - "
            f"Val PSNR: {val_metrics['val_psnr']:.2f}"
        )

        # 保存checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.num_epochs:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1:03d}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.tokenizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['train_loss'],
                'val_mse': val_metrics['val_mse'],
            }, checkpoint_path)
            logger.info(f"Checkpoint保存到: {checkpoint_path}")

        # 保存最佳模型
        if val_metrics['val_mse'] < best_val_loss:
            best_val_loss = val_metrics['val_mse']
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(model.tokenizer.state_dict(), best_path)

    logger.info("训练完成！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TiTok MRI Fine-tuning')
    parser.add_argument('--data_root', type=str,
                      default='/root/Documents/ICLR-Med/MedCompression/dataloader/acdc_img_datasets',
                      help='ACDC数据集路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='输出目录')
    parser.add_argument('--tokenizer_path', type=str,
                      default='./checkpoints/tokenizer_titok_b64',
                      help='Tokenizer路径')
    parser.add_argument('--generator_path', type=str,
                      default='./checkpoints/generator_titok_b64',
                      help='Generator路径')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--save_every', type=int, default=5,
                      help='每多少个epoch保存一次')
    parser.add_argument('--device', type=str, default='cuda',
                      help='计算设备')

    args = parser.parse_args()
    main(args)
