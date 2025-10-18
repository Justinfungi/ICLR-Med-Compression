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
import csv

# Manual implementations of image quality metrics
def compute_ssim(img1, img2, data_range=1.0, win_size=11, sigma=1.5):
    """Compute SSIM manually using torch operations"""
    try:
        # Create Gaussian kernel
        coords = torch.arange(win_size, dtype=torch.float32, device=img1.device)
        coords -= win_size // 2

        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()

        kernel = g[:, None] * g[None, :]
        kernel = kernel.expand(img1.shape[1], 1, win_size, win_size).contiguous()

        # Compute local means
        mu1 = torch.nn.functional.conv2d(img1, kernel, groups=img1.shape[1], padding=win_size//2)
        mu2 = torch.nn.functional.conv2d(img2, kernel, groups=img2.shape[1], padding=win_size//2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = torch.nn.functional.conv2d(img1 ** 2, kernel, groups=img1.shape[1], padding=win_size//2) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(img2 ** 2, kernel, groups=img2.shape[1], padding=win_size//2) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(img1 * img2, kernel, groups=img1.shape[1], padding=win_size//2) - mu1_mu2

        # Constants
        C1 = (0.01 * data_range) ** 2
        C2 = (0.03 * data_range) ** 2

        # Compute SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

        ssim_map = numerator / denominator
        return ssim_map.mean().item()

    except Exception as e:
        print(f"⚠️ Error computing SSIM: {e}")
        return None


# Try to import torchmetrics, fallback to manual implementations
try:
    from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
    from torchmetrics.image import PeakSignalNoiseRatio as PSNR
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as LPIPS
    HAS_TORCHMETRICS = True
    print("✅ torchmetrics available - using optimized implementations")
except ImportError:
    print("⚠️ torchmetrics not available - using manual implementations for SSIM")
    print("💡 SSIM will be computed manually, LPIPS will be skipped")
    HAS_TORCHMETRICS = False

# 添加项目路径 - 支持从med_mri目录或MedCompression目录运行
current_dir = Path(__file__).parent
med_root = current_dir if current_dir.name == 'med_mri' else Path.cwd()
med_compression_root = med_root.parent if med_root.name == 'med_mri' else med_root

sys.path.insert(0, str(med_compression_root))
sys.path.insert(0, str(med_root))

# 导入必要的模块
try:
    from modeling.titok import TiTok
    from modeling.maskgit import ImageBert
except ImportError:
    # 如果在1d-tokenizer中运行，调整路径
    sys.path.insert(0, str(med_compression_root / '1d-tokenizer'))
    from modeling.titok import TiTok
    from modeling.maskgit import ImageBert

try:
    from med_mri.acdc_dataset import create_data_loaders, ACDCMRIDataset
except ImportError:
    from acdc_dataset import create_data_loaders, ACDCMRIDataset

# 导入损失函数
try:
    from med_mri.loss import SimpleMRILoss, TiTokMRILoss
except ImportError:
    from loss import SimpleMRILoss, TiTokMRILoss

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
        checkpoint_path_obj = Path(checkpoint_path)

        # Check if local checkpoint directory exists and has model files
        if checkpoint_path_obj.exists() and checkpoint_path_obj.is_dir():
            model_files = list(checkpoint_path_obj.glob("*.safetensors")) + list(checkpoint_path_obj.glob("*.bin"))
            if model_files:
                print(f"✅ 从本地路径加载tokenizer: {checkpoint_path}")
                tokenizer = TiTok.from_pretrained(checkpoint_path)
                return tokenizer

        # Fallback to HuggingFace loading
        model_name = self._get_model_name_from_path(checkpoint_path)
        try:
            tokenizer = TiTok.from_pretrained(model_name)
            print(f"✅ 从HuggingFace加载tokenizer: {model_name}")
        except Exception as e:
            print(f"⚠️ HuggingFace加载失败: {e}")
            print(f"💡 请确保模型已下载到: {checkpoint_path}")
            print(f"   或运行: python download_checkpoints.py --best-only")
            raise e

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
        path_lower = path.lower()

        # Handle direct model names
        if path.startswith('yucornetto/'):
            return path

        # Handle local directory names
        if 'tokenizer_titok_bl128_vae_c16_imagenet' in path_lower:
            return 'yucornetto/tokenizer_titok_bl128_vae_c16_imagenet'
        elif 'tokenizer_titok_b64_imagenet' in path_lower:
            return 'yucornetto/tokenizer_titok_b64_imagenet'
        elif 'generator_titok_b64_imagenet' in path_lower:
            return 'yucornetto/generator_titok_b64_imagenet'
        elif 'b64' in path_lower:
            return 'yucornetto/tokenizer_titok_b64_imagenet'
        elif 'b128' in path_lower:
            return 'yucornetto/tokenizer_titok_b128_imagenet'

        return 'yucornetto/tokenizer_titok_b64_imagenet'

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播: 编码 -> 解码"""
        # 编码为token (returns (tokens, info_dict))
        encode_result = self.tokenizer.encode(x)
        if isinstance(encode_result, tuple):
            tokens, _ = encode_result  # tokens shape: [B, C, 1, 64]
        else:
            tokens = encode_result

        # 解码重建图像
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

    def __init__(self, device='cuda'):
        self.device = device
        if HAS_TORCHMETRICS:
            # Initialize metrics
            self.ssim = SSIM(data_range=1.0).to(device)
            self.psnr_metric = PSNR(data_range=1.0).to(device)
            self.lpips = LPIPS(net_type='alex').to(device)  # Using AlexNet backbone
        else:
            self.ssim = None
            self.psnr_metric = None
            self.lpips = None

    def compute_metrics(self, images, reconstructed, tokens):
        """计算重建指标"""
        # MSE
        mse = torch.mean((images - reconstructed) ** 2).item()

        # PSNR (manual calculation)
        max_val = 1.0
        psnr = 20 * np.log10(max_val / np.sqrt(mse)) if mse > 0 else 100.0

        # 压缩率
        original_pixels = images.shape[-2] * images.shape[-1] * images.shape[-3]
        n_tokens = tokens.shape[-1]
        compression_ratio = original_pixels / n_tokens

        metrics = {
            'mse': mse,
            'psnr': psnr,
            'compression_ratio': compression_ratio
        }

        # Additional metrics
        if HAS_TORCHMETRICS and self.ssim is not None:
            try:
                # SSIM (torchmetrics)
                ssim_val = self.ssim(reconstructed, images).item()
                metrics['ssim'] = ssim_val

                # PSNR (torchmetrics version)
                psnr_val = self.psnr_metric(reconstructed, images).item()
                metrics['psnr_torchmetrics'] = psnr_val

                # LPIPS
                lpips_val = self.lpips(reconstructed, images).mean().item()
                metrics['lpips'] = lpips_val

            except Exception as e:
                print(f"⚠️ Error computing torchmetrics: {e}")
        else:
            # Manual SSIM implementation when torchmetrics not available
            try:
                ssim_val = compute_ssim(reconstructed, images, data_range=1.0)
                if ssim_val is not None:
                    metrics['ssim'] = ssim_val
                    print(f"✅ SSIM computed manually: {ssim_val:.4f}")
                else:
                    print("⚠️ Failed to compute SSIM manually")
            except Exception as e:
                print(f"⚠️ Error computing manual SSIM: {e}")

            print("💡 LPIPS requires torchmetrics - skipping for now")

        return metrics


def save_sample_images(images, reconstructed, save_dir, epoch, prefix="val", max_samples=8):
    """
    保存输入和重建图像样本到epoch子文件夹

    Args:
        images: 原始图像 [B, C, H, W]
        reconstructed: 重建图像 [B, C, H, W]
        save_dir: 基础保存目录
        epoch: 当前epoch
        prefix: 文件名前缀 ("val" 或 "test")
        max_samples: 最大保存样本数
    """
    # 创建epoch子文件夹
    epoch_dir = Path(save_dir) / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # 限制样本数量
    n_samples = min(max_samples, images.shape[0])

    for i in range(n_samples):
        # 转换tensor到PIL图像
        # 输入图像 (假设是RGB格式，范围[0,1])
        input_img = images[i].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        input_img = (input_img * 255).astype(np.uint8)
        input_pil = Image.fromarray(input_img)

        # 重建图像
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        recon_img = np.clip(recon_img, 0, 1)  # 确保范围在[0,1]
        recon_img = (recon_img * 255).astype(np.uint8)
        recon_pil = Image.fromarray(recon_img)

        # 保存图像到epoch文件夹
        input_path = epoch_dir / f"{prefix}_sample_{i:02d}_input.png"
        recon_path = epoch_dir / f"{prefix}_sample_{i:02d}_recon.png"

        input_pil.save(input_path)
        recon_pil.save(recon_path)


def init_metrics_csv(csv_path):
    """初始化metrics CSV文件"""
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头 - 扩展支持新的损失组件
        headers = [
            'epoch', 'phase', 'total_loss', 'reconstruction_loss', 'perceptual_loss',
            'weighted_gan_loss', 'discriminator_loss', 'logits_real', 'logits_fake',
            'mse', 'psnr', 'compression_ratio', 'ssim'
        ]
        if HAS_TORCHMETRICS:
            headers.extend(['psnr_torchmetrics', 'lpips'])
        writer.writerow(headers)


def log_metrics_to_csv(csv_path, epoch, phase, metrics):
    """将metrics记录到CSV文件"""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Handle different key names for different phases
        if phase == 'train':
            # Training has loss components but no validation metrics
            total_loss = metrics.get('total_loss', '')
            reconstruction_loss = metrics.get('reconstruction_loss', '')
            perceptual_loss = metrics.get('perceptual_loss', '')
            weighted_gan_loss = metrics.get('weighted_gan_loss', '')
            discriminator_loss = metrics.get('discriminator_loss', '')
            logits_real = metrics.get('logits_real', '')
            logits_fake = metrics.get('logits_fake', '')
            mse = psnr = compression_ratio = ssim = ''
        else:
            # Validation/Test have evaluation metrics
            total_loss = reconstruction_loss = perceptual_loss = weighted_gan_loss = ''
            discriminator_loss = logits_real = logits_fake = ''
            mse = metrics.get('val_mse', metrics.get('mse', ''))
            psnr = metrics.get('val_psnr', metrics.get('psnr', ''))
            compression_ratio = metrics.get('compression_ratio', '')
            ssim = metrics.get('ssim', '')

        row = [
            epoch, phase, total_loss, reconstruction_loss, perceptual_loss,
            weighted_gan_loss, discriminator_loss, logits_real, logits_fake,
            mse, psnr, compression_ratio, ssim
        ]

        if HAS_TORCHMETRICS:
            row.extend([metrics.get('psnr_torchmetrics', ''), metrics.get('lpips', '')])

        writer.writerow(row)


def train_one_epoch(
    model: TiTokMRIWrapper,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    epoch: int,
    total_epochs: int,
    global_step: int = 0,
    discriminator_optimizer: optim.Optimizer = None,
) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()  # 确保模型处于训练模式
    model.tokenizer.train()  # 确保tokenizer处于训练模式

    total_metrics = {}
    n_batches = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]")

    for batch in pbar:
        images = batch['image'].to(device)

        # 前向传播
        reconstructed, tokens = model(images)

        # 计算生成器损失 (始终训练)
        loss, loss_dict = loss_fn(images, reconstructed, global_step, mode="generator")

        # 反向传播 (生成器)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累积生成器metrics
        for key, value in loss_dict.items():
            if key not in total_metrics:
                total_metrics[key] = 0.0
            total_metrics[key] += value.item()

        # 训练判别器 (如果启用且到达训练步数)
        if discriminator_optimizer is not None and hasattr(loss_fn, 'should_discriminator_be_trained') and loss_fn.should_discriminator_be_trained(global_step):
            # 判别器训练步
            discriminator_loss, discriminator_loss_dict = loss_fn(images, reconstructed, global_step, mode="discriminator")

            # 反向传播 (判别器)
            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # 累积判别器metrics
            for key, value in discriminator_loss_dict.items():
                metric_key = f"discriminator_{key}"
                if metric_key not in total_metrics:
                    total_metrics[metric_key] = 0.0
                total_metrics[metric_key] += value.item() if torch.is_tensor(value) else value

        n_batches += 1
        global_step += 1

        # 更新进度条
        pbar_metrics = {'total_loss': total_metrics['total_loss'] / n_batches}
        if 'reconstruction_loss' in total_metrics:
            pbar_metrics['recon_loss'] = total_metrics['reconstruction_loss'] / n_batches
        if 'discriminator_loss' in total_metrics:
            pbar_metrics['disc_loss'] = total_metrics['discriminator_loss'] / n_batches
        pbar.set_postfix(pbar_metrics)

    # 计算平均值
    avg_metrics = {key: value / n_batches for key, value in total_metrics.items()}
    return avg_metrics


def validate(
    model: TiTokMRIWrapper,
    val_loader: DataLoader,
    device: str,
    epoch: int,
    evaluator: TiTokMRIEvaluator,
    save_images: bool = False,
    save_dir: str = None,
    prefix: str = "val"
) -> Dict[str, float]:
    """验证"""
    model.eval()
    total_mse = 0.0
    total_psnr = 0.0
    total_compression_ratio = 0.0
    total_ssim = 0.0
    total_psnr_torchmetrics = 0.0
    total_lpips = 0.0
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Validating")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            reconstructed, tokens = model(images)

            metrics = evaluator.compute_metrics(images, reconstructed, tokens)

            # Accumulate all metrics
            total_mse += metrics['mse']
            total_psnr += metrics['psnr']
            total_compression_ratio += metrics['compression_ratio']

            if 'ssim' in metrics:
                total_ssim += metrics['ssim']
            if 'psnr_torchmetrics' in metrics:
                total_psnr_torchmetrics += metrics['psnr_torchmetrics']
            if 'lpips' in metrics:
                total_lpips += metrics['lpips']

            n_batches += 1

            # 保存第一批的图像样本
            if save_images and batch_idx == 0 and save_dir is not None:
                save_sample_images(images, reconstructed, save_dir, epoch, prefix)

            # Update progress bar with available metrics
            pbar_metrics = {'mse': total_mse / n_batches, 'psnr': total_psnr / n_batches}
            if 'ssim' in metrics:
                pbar_metrics['ssim'] = total_ssim / n_batches
            pbar.set_postfix(pbar_metrics)

    # Build return dictionary with all available metrics
    result = {
        'val_mse': total_mse / n_batches,
        'val_psnr': total_psnr / n_batches,
        'compression_ratio': total_compression_ratio / n_batches
    }

    # Add additional metrics if available
    if total_ssim > 0:
        result['ssim'] = total_ssim / n_batches
    if total_psnr_torchmetrics > 0:
        result['psnr_torchmetrics'] = total_psnr_torchmetrics / n_batches
    if total_lpips > 0:
        result['lpips'] = total_lpips / n_batches

    return result


def main(args):
    """主训练函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # 设备 - 优先使用CUDA
    if args.device == 'cuda' and torch.cuda.is_available():
        device = 'cuda'
        logger.info(f"使用CUDA设备: {torch.cuda.get_device_name()}")
    elif args.device == 'cpu':
        device = 'cpu'
        logger.info("使用CPU设备")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"自动选择设备: {device}")

    if device == 'cpu':
        logger.warning("⚠️ 使用CPU训练将非常慢，建议使用CUDA GPU")

    # 创建带时间戳的运行目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"run_{timestamp}"

    # 输出目录结构
    output_dir = Path(args.output_dir)
    run_dir = output_dir / 'checkpoints' / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # 子目录
    checkpoint_dir = run_dir  # checkpoints直接保存在run目录下
    images_base_dir = run_dir / 'images'
    logs_dir = run_dir / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件
    metrics_log_path = logs_dir / 'training_metrics.csv'
    training_log_path = logs_dir / 'training.log'

    # 设置日志同时输出到文件和控制台
    file_handler = logging.FileHandler(training_log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    logger.info(f"运行目录: {run_dir}")
    logger.info(f"检查点保存到: {checkpoint_dir}")
    logger.info(f"日志保存到: {logs_dir}")

    # 图像保存目录 (按epoch组织)
    images_dir = images_base_dir if args.save_images else None
    if images_dir:
        logger.info(f"图像将保存到: {images_dir} (按epoch组织)")

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

    # 确保tokenizer参数可训练
    for param in model.tokenizer.parameters():
        param.requires_grad = True

    # 调试：检查可训练参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")

    # 创建损失函数
    logger.info("创建损失函数...")
    loss_config = {
        'reconstruction_weight': getattr(args, 'reconstruction_weight', 1.0),
        'perceptual_weight': getattr(args, 'perceptual_weight', 0.1),
        'reconstruction_loss_type': getattr(args, 'reconstruction_loss_type', 'l2'),
        'perceptual_net_type': getattr(args, 'perceptual_net_type', 'vgg16'),
        'use_gan': getattr(args, 'use_gan', False),
        'discriminator_weight': getattr(args, 'discriminator_weight', 0.5),
        'discriminator_start': getattr(args, 'discriminator_start', 1000),
        'discriminator_input_nc': 3,  # TiTok输出RGB图像
        'discriminator_ndf': 32,
        'discriminator_layers': 3,
    }

    if getattr(args, 'use_full_loss', False):
        loss_fn = TiTokMRILoss(loss_config)
        logger.info("使用完整的TiTok MRI损失函数")
    else:
        loss_fn = SimpleMRILoss(loss_config)
        logger.info("使用简化的MRI损失函数 (重建 + 感知)")

    loss_fn = loss_fn.to(device)

    # 创建判别器优化器 (如果使用GAN)
    discriminator_optimizer = None
    if hasattr(loss_fn, 'discriminator') and loss_fn.use_gan:
        import torch.optim as optim
        discriminator_params = list(loss_fn.discriminator.parameters())
        discriminator_optimizer = optim.AdamW(discriminator_params, lr=1e-4, weight_decay=1e-4)
        logger.info("创建判别器优化器")

    # 优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # 评估器
    evaluator = TiTokMRIEvaluator(device=device)

    # 初始化metrics CSV
    init_metrics_csv(metrics_log_path)
    logger.info(f"Metrics将记录到: {metrics_log_path}")

    # 训练循环
    best_val_loss = float('inf')
    global_step = 0
    logger.info("开始训练...")

    for epoch in range(args.num_epochs):
        # 训练
        train_metrics = train_one_epoch(
            model,
            data_loaders['train'],
            optimizer,
            loss_fn,
            device,
            epoch,
            args.num_epochs,
            global_step,
            discriminator_optimizer
        )

        # 更新global_step
        global_step += len(data_loaders['train'])

        # 验证 - 定期保存图像样本
        save_images_current = args.save_images and ((epoch + 1) % args.save_image_every == 0 or epoch == 0)
        val_metrics = validate(
            model,
            data_loaders['val'],
            device,
            epoch,
            evaluator,
            save_images=save_images_current,
            save_dir=str(images_dir) if images_dir else None,
            prefix="val"
        )

        scheduler.step()

        # 记录metrics到CSV
        log_metrics_to_csv(metrics_log_path, epoch + 1, 'train', train_metrics)
        log_metrics_to_csv(metrics_log_path, epoch + 1, 'val', val_metrics)

        # 日志 - 使用新的metrics结构
        train_loss_display = train_metrics.get('total_loss', train_metrics.get('train_loss', 0.0))
        log_msg = f"Epoch {epoch + 1}/{args.num_epochs} - Train Loss: {train_loss_display:.6f} - Val MSE: {val_metrics['val_mse']:.6f} - Val PSNR: {val_metrics['val_psnr']:.2f}"

        # 添加训练损失组件详情
        if 'reconstruction_loss' in train_metrics:
            log_msg += f" - Recon: {train_metrics['reconstruction_loss']:.6f}"
        if 'perceptual_loss' in train_metrics:
            log_msg += f" - Percep: {train_metrics['perceptual_loss']:.6f}"

        # 添加验证metrics详情
        if 'ssim' in val_metrics:
            log_msg += f" - Val SSIM: {val_metrics['ssim']:.4f}"
        if 'lpips' in val_metrics:
            log_msg += f" - Val LPIPS: {val_metrics['lpips']:.4f}"

        logger.info(log_msg)

        # 保存checkpoint
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.num_epochs:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch + 1:03d}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.tokenizer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_metrics['total_loss'],
                'val_mse': val_metrics['val_mse'],
            }, checkpoint_path)
            logger.info(f"Checkpoint保存到: {checkpoint_path}")

        # 保存最佳模型
        if val_metrics['val_mse'] < best_val_loss:
            best_val_loss = val_metrics['val_mse']
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(model.tokenizer.state_dict(), best_path)

    # 测试集评估
    logger.info("在测试集上进行最终评估...")
    test_metrics = validate(
        model,
        data_loaders['test'],
        device,
        epoch=-1,  # 使用-1表示测试
        evaluator=evaluator,
        save_images=args.save_images,
        save_dir=str(images_dir) if images_dir else None,
        prefix="test"
    )

    # 记录测试metrics到CSV
    log_metrics_to_csv(metrics_log_path, -1, 'test', test_metrics)

    # 测试结果日志
    test_log_msg = f"测试结果 - MSE: {test_metrics['val_mse']:.6f} - PSNR: {test_metrics['val_psnr']:.2f}"

    # 添加额外metrics到测试日志
    if 'ssim' in test_metrics:
        test_log_msg += f" - SSIM: {test_metrics['ssim']:.4f}"
    if 'lpips' in test_metrics:
        test_log_msg += f" - LPIPS: {test_metrics['lpips']:.4f}"

    logger.info(test_log_msg)
    logger.info("训练完成！")
    logger.info(f"所有输出保存在: {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TiTok MRI Fine-tuning')
    parser.add_argument('--data_root', type=str,
                      default='../acdc_img_datasets',
                      help='ACDC数据集路径')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                      help='输出目录')
    parser.add_argument('--tokenizer_path', type=str,
                      default='./checkpoints/tokenizer_titok_bl128_vae_c16_imagenet',
                      help='Tokenizer checkpoint路径 (默认: 最佳性能模型)')
    parser.add_argument('--generator_path', type=str,
                      default=None,
                      help='Generator checkpoint路径 (可选)')
    parser.add_argument('--batch_size', type=int, default=8,
                      help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=20,
                      help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='学习率')
    parser.add_argument('--save_every', type=int, default=5,
                      help='每多少个epoch保存一次checkpoint')
    parser.add_argument('--save_images', action='store_true',
                      help='保存验证和测试图像样本')
    parser.add_argument('--save_image_every', type=int, default=5,
                      help='每多少个epoch保存一次验证图像')
    parser.add_argument('--device', type=str, default='cuda',
                      help='计算设备 (cuda/cpu/auto)')

    # 损失函数相关参数
    parser.add_argument('--use_full_loss', action='store_true',
                      help='使用完整的TiTok损失函数 (包含GAN选项，默认使用简化版本)')
    parser.add_argument('--reconstruction_weight', type=float, default=1.0,
                      help='重建损失权重')
    parser.add_argument('--perceptual_weight', type=float, default=0.1,
                      help='感知损失权重')
    parser.add_argument('--reconstruction_loss_type', type=str, default='l2',
                      choices=['l1', 'l2'], help='重建损失类型 (l1或l2)')
    parser.add_argument('--perceptual_net_type', type=str, default='vgg16',
                      choices=['vgg16', 'vgg19'], help='感知损失使用的网络类型')
    parser.add_argument('--use_gan', action='store_true',
                      help='启用GAN对抗训练 (仅在使用完整损失函数时有效)')
    parser.add_argument('--discriminator_weight', type=float, default=0.5,
                      help='判别器损失权重')
    parser.add_argument('--discriminator_start', type=int, default=1000,
                      help='开始GAN训练的步数')

    args = parser.parse_args()
    main(args)
