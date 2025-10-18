# 🫀 TiTok MRI 微调

在ACDC心脏MRI数据集上微调TiTok模型，用于医疗图像压缩。

**特性**：~3000x压缩比，CPU/GPU支持，PyTorch实现。

## 📊 数据集设置

### 转换ACDC数据集 (NIfTI → PNG)

如果您有原始ACDC数据集的NIfTI格式，请转换为PNG图像：

```bash
# 导航到med_mri目录
cd med_mri

# 转换NIfTI文件为PNG (需要nibabel)
python convert_acdc.py

# 预期输出结构：
# acdc_img_datasets/
# ├── patient001/
# │   ├── frame_000_slice_000.png
# │   ├── frame_000_slice_001.png
# │   └── ...
# ├── patient002/
# │   └── ...
# └── ...
```

**要求**：安装nibabel用于NIfTI文件处理
```bash
pip install nibabel
```

**来源**：将原始ACDC数据集放置在 `./acdc_dataset/` (相对于med_mri目录)

### 预转换数据集

确保ACDC数据集可用：
```
../acdc_img_datasets/
```

## 🚀 命令

### 安装依赖
```bash
cd /userhome/cs3/fung0311/CVPR-2025/MedCompression
pip install torch torchvision transformers pillow numpy matplotlib tqdm pyyaml omegaconf
```

### 下载检查点
```bash
# 🚀 自动下载最佳性能模型 (H800推荐)
python download_checkpoints.py --best-only

# 交互式模型选择 (显示带有推荐的菜单)
python download_checkpoints.py

# 下载最佳性能模型 (非交互式)
python download_checkpoints.py --non-interactive

# 下载到自定义目录
python download_checkpoints.py --output_dir ../my_checkpoints --best-only

# 下载特定模型
python download_checkpoints.py --models tokenizer_bl128_vae  # 最佳FID (0.84)
python download_checkpoints.py --models tokenizer_b64        # 平衡性能
python download_checkpoints.py --models tokenizer_bl128_vq   # VQ替代方案

# 列出所有可用模型
python download_checkpoints.py --list

**⚠️ 需要认证：**
```bash
# 这些TiTok模型需要HuggingFace认证
huggingface-cli login

# 或者设置环境变量：
# export HF_TOKEN=your_token_here
```

**手动下载：** 如果自动下载失败，请手动从以下链接下载：
- https://huggingface.co/yucornetto/tokenizer_titok_bl128_vae_c16_imagenet
- https://huggingface.co/yucornetto/tokenizer_titok_b64_imagenet
```

**模型性能 (FID分数 - 越低越好)：**
- `tokenizer_bl128_vae` - **FID: 0.84** ⭐ 最佳性能 (H800默认)
- `tokenizer_sl256_vq` - FID: 1.03
- `tokenizer_bl128_vq` - FID: 1.49
- `tokenizer_b64` - FID: 1.70 (平衡大小/性能)
- `tokenizer_bl64_vae` - FID: 1.25
- `tokenizer_ll32_vae` - FID: 1.61

**可用模型：**
- **VAE模型**：最佳重建质量，适合H800微调 ⭐
- **VQ模型**：质量和速度的良好平衡
- **标准模型**：更小尺寸，更快训练

## 🔄 TiTok组件说明

### **Tokenizer (必需)**
- **目的**：将图像压缩为离散token (256×256图像压缩为32个token)
- **使用方法**：`tokenizer.encode(image)` → tokens, `tokenizer.decode(tokens)` → 重建图像
- **我们的用途**：医疗图像压缩用于MRI数据

### **Generator (可选)**
- **目的**：从头生成新图像使用学习的token模式
- **使用方法**：创建新颖图像，不是重建
- **我们的用途**：医疗压缩任务不需要

### **为什么不需要Generator**
- ✅ **仅重建**：我们只需要压缩/解压
- ✅ **医疗重点**：诊断/压缩不需要生成
- ✅ **训练更快**：更少的参数需要训练
- ✅ **H800优化**：更少的内存使用

### 测试设置
```bash
cd med_mri
python test_finetune.py
```

### 训练模型 (CUDA - 推荐)
```bash
# 基础CUDA训练 (默认简化损失函数)
python finetune_titok_mri.py --batch_size 8 --num_epochs 20

# CUDA训练并保存图像
python finetune_titok_mri.py --batch_size 8 --num_epochs 20 --save_images --save_image_every 5

# 🚀 完整示例 - 所有功能开启 (推荐用于高质量重建)
python finetune_titok_mri.py \
  --data_root ../acdc_img_datasets \
  --output_dir ./outputs \
  --tokenizer_path ./checkpoints/tokenizer_titok_bl128_vae_c16_imagenet \
  --batch_size 8 \
  --num_epochs 50 \
  --learning_rate 1e-4 \
  --save_every 10 \
  --save_images \
  --save_image_every 5 \
  --device cuda \
  --use_full_loss \
  --use_gan \
  --reconstruction_weight 1.0 \
  --perceptual_weight 0.15 \
  --reconstruction_loss_type l2 \
  --perceptual_net_type vgg16 \
  --discriminator_weight 0.8 \
  --discriminator_start 5

# 🎯 高质量重建配置 (平衡速度和质量)
python finetune_titok_mri.py \
  --batch_size 16 \
  --num_epochs 30 \
  --perceptual_weight 0.2 \
  --reconstruction_loss_type l1 \
  --save_images \
  --save_image_every 10

# 🏃‍♂️ 快速实验配置 (用于调试)
python finetune_titok_mri.py \
  --batch_size 4 \
  --num_epochs 5 \
  --perceptual_weight 0.05 \
  --save_images \
  --save_image_every 1
```

### 通过SLURM训练模型 (GPU)
```bash
./gpu_run.sh train
```

### 训练模型 (CPU - 慢)
```bash
python finetune_titok_mri.py --batch_size 2 --num_epochs 5 --device cpu
```

### GPU交互式会话
```bash
./gpu_run.sh bash
```

## ⚙️ 配置

### 📋 完整参数列表

#### 基础训练参数
- `--data_root`：ACDC数据集路径 (默认: `../acdc_img_datasets`)
- `--output_dir`：输出目录 (默认: `./outputs`)
- `--tokenizer_path`：Tokenizer checkpoint路径
- `--batch_size`：训练批次大小 (默认: 8，建议: 4-16)
- `--num_epochs`：训练轮数 (默认: 20，建议: 20-100)
- `--learning_rate`：学习率 (默认: 1e-4，建议: 1e-5 到 5e-4)
- `--device`：计算设备 (`cuda`/`cpu`/auto，默认: `cuda`)

#### 检查点和保存参数
- `--save_every`：每N个epoch保存checkpoint (默认: 5)
- `--save_images`：保存验证/测试图像样本 (标志)
- `--save_image_every`：每N个epoch保存验证图像 (默认: 5)

#### 🎯 损失函数参数

##### 简化的MRI损失函数 (默认)
- `--reconstruction_weight`：重建损失权重 (默认: 1.0)
- `--perceptual_weight`：感知损失权重 (默认: 0.1，建议: 0.05-0.2)
- `--reconstruction_loss_type`：重建损失类型 (`l1`/`l2`，默认: `l2`)
- `--perceptual_net_type`：感知损失网络 (`vgg16`/`vgg19`，默认: `vgg16`)

##### 完整的TiTok损失函数
- `--use_full_loss`：启用完整的TiTok损失函数 (标志)
- `--use_gan`：启用GAN对抗训练 (标志)
- `--discriminator_weight`：判别器损失权重 (默认: 0.5，建议: 0.3-1.0)
- `--discriminator_start`：GAN训练开始步数 (默认: 1000，建议: 500-2000)

### 💡 使用建议

#### 🎯 高质量重建 (推荐用于最终模型)
```bash
# 使用完整损失函数，较高的感知权重
--use_full_loss --use_gan --perceptual_weight 0.15 --discriminator_weight 0.8
```

#### ⚡ 快速训练 (用于实验和调试)
```bash
# 简化损失，较小的batch size
--batch_size 4 --num_epochs 10 --perceptual_weight 0.05
```

#### 🏥 医学应用 (平衡质量和稳定性)
```bash
# L1损失更稳定，适中的感知权重
--reconstruction_loss_type l1 --perceptual_weight 0.1 --batch_size 8
```

#### 🚀 大规模训练 (需要大量GPU资源)
```bash
# 更大的batch size，更长的训练
--batch_size 32 --num_epochs 100 --perceptual_weight 0.2 --use_full_loss
```


## 📁 输出目录结构

```
outputs/checkpoints/run_<YYYYMMDD_HHMMSS>/
├── checkpoint_epoch_005.pt          # 检查点
├── best_model.pt                    # 最佳模型
├── images/                          # 图像保存
│   ├── epoch_000/                   # 第0个epoch
│   │   ├── val_sample_00_input.png
│   │   ├── val_sample_00_recon.png
│   │   └── ...
│   ├── epoch_005/                   # 第5个epoch
│   │   └── ...
│   └── epoch_010/                   # 第10个epoch
│       └── ...
└── logs/                            # 日志文件
    ├── training.log                 # 训练日志
    └── training_metrics.csv         # Metrics CSV
```

## 🔧 基本故障排除

**CUDA不可用**：使用 `--device cpu`
```bash
python finetune_titok_mri.py --device cpu
```

**内存不足**：减少批次大小
```bash
python finetune_titok_mri.py --batch_size 2
```

**数据集未找到**：检查路径
```bash
ls ../acdc_img_datasets/
```
