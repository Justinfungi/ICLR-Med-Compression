#!/usr/bin/env python3
"""
Standalone training script - Run from med_mri directory
python run_train.py --batch_size 4 --num_epochs 5 --device cpu
"""

import sys
import argparse
from pathlib import Path

# Add med_mri directory to path
med_mri_dir = Path(__file__).parent
sys.path.insert(0, str(med_mri_dir))
sys.path.insert(0, str(med_mri_dir.parent))

# Import training components
from finetune_titok_mri import (
    TiTokMRIWrapper,
    TiTokMRIEvaluator,
    train_one_epoch,
    validate,
    main
)
from acdc_dataset import create_data_loaders

import torch
import torch.optim as optim
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


def train(args):
    """Enhanced training function"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Device
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    logger.info(f"使用设备: {device}")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"输出目录: {output_dir}")

    # Create data loaders
    logger.info("创建数据加载器...")
    data_loaders = create_data_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=4 if device == 'cuda' else 0,
        image_size=(256, 256),
        augment=True
    )

    # Create model
    logger.info("创建TiTok模型...")
    model = TiTokMRIWrapper(
        tokenizer_path=args.tokenizer_path,
        generator_path=args.generator_path,
        device=device
    ).to(device)

    # Freeze generator if specified
    if model.generator is not None and args.freeze_generator:
        for param in model.generator.parameters():
            param.requires_grad = False
        logger.info("已冻结Generator参数")

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # Evaluator
    evaluator = TiTokMRIEvaluator()

    # Training loop
    best_val_loss = float('inf')
    logger.info("开始训练...")

    for epoch in range(args.num_epochs):
        # Train
        train_metrics = train_one_epoch(
            model,
            data_loaders['train'],
            optimizer,
            device,
            epoch,
            args.num_epochs
        )

        # Validate
        val_metrics = validate(
            model,
            data_loaders['val'],
            device,
            epoch,
            evaluator
        )

        scheduler.step()

        # Log
        logger.info(
            f"Epoch {epoch + 1}/{args.num_epochs} - "
            f"Train Loss: {train_metrics['train_loss']:.6f} - "
            f"Val MSE: {val_metrics['val_mse']:.6f} - "
            f"Val PSNR: {val_metrics['val_psnr']:.2f}"
        )

        # Save checkpoint
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

        # Save best model
        if val_metrics['val_mse'] < best_val_loss:
            best_val_loss = val_metrics['val_mse']
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save(model.tokenizer.state_dict(), best_path)
            logger.info(f"最佳模型已保存")

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
                      help='计算设备 (cuda 或 cpu)')
    parser.add_argument('--freeze_generator', action='store_true', default=True,
                      help='冻结Generator参数')

    args = parser.parse_args()
    train(args)
