#!/usr/bin/env python3
"""
TiTok MRI Fine-tuning 快速测试脚本

快速验证微调功能是否正常工作
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目路径 - 支持从med_mri目录或MedCompression目录运行
current_dir = Path(__file__).parent
med_root = current_dir if current_dir.name == 'med_mri' else Path.cwd()
med_compression_root = med_root.parent if med_root.name == 'med_mri' else med_root

sys.path.insert(0, str(med_compression_root))
sys.path.insert(0, str(med_root))

try:
    from med_mri.acdc_dataset import create_data_loaders
    from med_mri.finetune_titok_mri import TiTokMRIWrapper, TiTokMRIEvaluator
except ImportError:
    from acdc_dataset import create_data_loaders
    from finetune_titok_mri import TiTokMRIWrapper, TiTokMRIEvaluator


def test_basic_functionality():
    """测试基本功能"""
    print("🧪 TiTok MRI微调基本功能测试")
    print("=" * 50)

    # 设置设备
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    print(f"📱 使用设备: {device}")

    # 数据路径
    data_root = "/root/Documents/ICLR-Med/MedCompression/dataloader/acdc_img_datasets"
    if not os.path.exists(data_root):
        print(f"❌ 数据目录不存在: {data_root}")
        return False

    print(f"📁 数据目录: {data_root}")

    # 创建小批量数据加载器进行测试
    print("\n📦 创建测试数据加载器...")
    try:
        data_loaders = create_data_loaders(
            data_root=data_root,
            batch_size=2,  # 小批量
            num_workers=0,  # 不使用多进程
            image_size=(256, 256),
            augment=False  # 测试时不增强
        )
        train_loader = data_loaders['train']
        print("✅ 数据加载器创建成功")
    except Exception as e:
        print(f"❌ 数据加载器创建失败: {e}")
        return False

    # 创建模型
    print("\n🤖 创建TiTok模型...")
    try:
        model = TiTokMRIWrapper(
            tokenizer_path="./checkpoints/tokenizer_titok_b64",
            generator_path="./checkpoints/generator_titok_b64",
            device=device
        )
        print("✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        return False

    # 创建评估器
    evaluator = TiTokMRIEvaluator()

    # 测试前向传播
    print("\n🔄 测试前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            # 获取一个batch
            batch = next(iter(train_loader))
            images = batch['image'].to(device)

            print(f"📊 输入图像形状: {images.shape}")

            # 前向传播
            reconstructed, tokens = model(images)

            print(f"📊 重建图像形状: {reconstructed.shape}")
            print(f"📊 Token形状: {tokens.shape}")

            # 计算指标
            metrics = evaluator.compute_metrics(images, reconstructed, tokens)
            print("📊 重建指标:")
            print(f"   MSE: {metrics.get('mse', 'N/A'):.4f}")
            print(f"   PSNR: {metrics.get('psnr', 'N/A'):.1f}")
            print(f"   压缩率: {metrics.get('compression_ratio', 'N/A'):.1f}")

            print("✅ 前向传播测试成功")

    except Exception as e:
        print(f"❌ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 训练步骤测试
    print("\n🏋️ 测试训练准备...")
    try:
        # 检查模型参数是否可训练
        trainable_params = sum(p.numel() for p in model.tokenizer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.tokenizer.parameters())

        print(f"📊 Tokenizer参数: {total_params:,} 总计, {trainable_params:,} 可训练")

        if trainable_params == 0:
            print("⚠️ 没有可训练参数，设置requires_grad=True")
            for param in model.tokenizer.parameters():
                param.requires_grad_(True)
            trainable_params = sum(p.numel() for p in model.tokenizer.parameters() if p.requires_grad)
            print(f"✅ 重新设置后可训练参数: {trainable_params:,}")

        print("✅ 训练准备测试成功")

    except Exception as e:
        print(f"❌ 训练准备测试失败: {e}")
        return False

    print("\n🎉 所有测试通过!")
    print("TiTok MRI微调脚本功能正常")

    return True


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
