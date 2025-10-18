# 🫀 TiTok MRI Fine-tuning Module

A comprehensive framework for fine-tuning the TiTok model on ACDC heart MRI dataset for medical image compression and reconstruction tasks.

## Overview

This module implements fine-tuning of the TiTok tokenizer on the ACDC (Automated Cardiac Diagnosis Challenge) MRI dataset. The TiTok model efficiently compresses cardiac MRI images into discrete tokens while maintaining reconstruction quality.

### Key Features

- **Efficient Compression**: ~3000x compression ratio with minimal quality loss
- **Medical Focus**: Optimized for cardiac MRI analysis
- **Flexible Architecture**: Supports both tokenizer and generator fine-tuning
- **Easy Integration**: Clean PyTorch implementation with HuggingFace model support
- **CPU & GPU Support**: Works on both CPU and CUDA devices

## 🏗️ Project Structure

```
med_mri/
├── acdc_dataset.py          # ACDC dataset loader and preprocessing
├── finetune_titok_mri.py    # Main fine-tuning script
├── test_finetune.py         # Quick functionality test
├── config_finetune.yaml     # Configuration file
└── README.md                # This file
```

## 📊 Dataset

### ACDC (Automated Cardiac Diagnosis Challenge)

- **Total Images**: 2,758 cardiac MRI frames
- **Patients**: 100 cardiac patients
- **Image Size**: 256×256 pixels (resized)
- **Dataset Split**:
  - **Training**: 70 patients, 2,029 images (71%)
  - **Validation**: 20 patients, 504 images (18%)
  - **Testing**: 10 patients, 225 images (11%)

### Data Format

- **Input**: PNG images from ACDC dataset
- **Format**: Cardiac MRI frames (grayscale, converted to RGB)
- **Resolution**: Variable → 256×256 (normalized)
- **Range**: [0, 1] (normalized)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd /root/Documents/ICLR-Med/MedCompression

# Install dependencies
pip install torch torchvision
pip install transformers huggingface-hub
pip install pillow numpy matplotlib tqdm pyyaml omegaconf
```

### 2. Prepare Dataset

Ensure ACDC dataset is available at:
```
/root/Documents/ICLR-Med/MedCompression/dataloader/acdc_img_datasets/
```

Dataset structure:
```
acdc_img_datasets/
├── patient001/
│   ├── frame_000.png
│   ├── frame_001.png
│   └── ...
├── patient002/
│   └── ...
└── ...
```

### 3. Test Installation

```bash
cd /root/Documents/ICLR-Med/MedCompression/1d-tokenizer

# Quick test
python ../med_mri/test_finetune.py
```

Expected output:
```
🎉 所有测试通过!
TiTok MRI微调脚本功能正常
```

### 4. Start Training

```bash
# CPU training (recommended for testing)
python ../med_mri/finetune_titok_mri.py \
    --batch_size 4 \
    --num_epochs 5 \
    --learning_rate 1e-4 \
    --device cpu

# GPU training (if available)
python ../med_mri/finetune_titok_mri.py \
    --batch_size 8 \
    --num_epochs 20 \
    --learning_rate 1e-4 \
    --device cuda
```

## 📝 Core Components

### 1. ACDCMRIDataset (`acdc_dataset.py`)

ACDC MRI dataset loader with support for train/val/test splits.

**Key Methods:**
- `__init__()`: Initialize dataset with split and preprocessing options
- `__len__()`: Return dataset size
- `__getitem__()`: Load and preprocess individual images

**Usage:**
```python
from med_mri.acdc_dataset import create_data_loaders

loaders = create_data_loaders(
    data_root="/path/to/acdc_img_datasets",
    batch_size=8,
    num_workers=4,
    image_size=(256, 256),
    augment=True
)

# Access loaders
train_loader = loaders['train']
val_loader = loaders['val']
test_loader = loaders['test']
```

### 2. TiTokMRIWrapper (`finetune_titok_mri.py`)

PyTorch module wrapper for TiTok model optimized for MRI tasks.

**Key Classes:**
- `TiTokMRIWrapper`: Main model wrapper
  - `forward()`: Encode image → reconstruct image
  - `tokenize()`: Only encode to tokens
  - `to()`: Move to device

- `TiTokMRIEvaluator`: Metrics computation
  - `compute_metrics()`: Calculate MSE, PSNR, compression ratio

**Usage:**
```python
from med_mri.finetune_titok_mri import TiTokMRIWrapper

model = TiTokMRIWrapper(
    tokenizer_path="./checkpoints/tokenizer_titok_b64",
    generator_path="./checkpoints/generator_titok_b64",
    device='cuda'
)

# Forward pass
reconstructed, tokens = model(images)  # images: [B, 3, 256, 256]
                                        # reconstructed: [B, 3, 256, 256]
                                        # tokens: [B, 12, 1, 64]
```

### 3. Training Functions

**train_one_epoch()**: Single epoch training
```python
metrics = train_one_epoch(
    model, train_loader, optimizer, device,
    epoch=0, total_epochs=20
)
```

**validate()**: Validation loop
```python
val_metrics = validate(
    model, val_loader, device, epoch=0,
    evaluator=evaluator
)
```

## ⚙️ Configuration

Edit `config_finetune.yaml` to customize training:

```yaml
# Data configuration
data:
  batch_size: 8
  image_size: [256, 256]
  augment: true
  num_workers: 4

# Training configuration
training:
  num_epochs: 20
  learning_rate: 1e-4
  weight_decay: 1e-4
  save_every: 5  # Save checkpoint every 5 epochs

# Model configuration
model:
  freeze_generator: true  # Only fine-tune tokenizer
  tokenizer_path: "./checkpoints/tokenizer_titok_b64"
  generator_path: "./checkpoints/generator_titok_b64"

# Device
device: "cuda"  # or "cpu"
```

## 🎯 Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_root` | ACDC path | Dataset root directory |
| `--output_dir` | `./outputs` | Output directory for checkpoints |
| `--batch_size` | 8 | Training batch size |
| `--num_epochs` | 20 | Total training epochs |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--save_every` | 5 | Save checkpoint every N epochs |
| `--device` | `cuda` | Device (`cuda` or `cpu`) |

## 📊 Expected Results

### Model Statistics

- **Total Parameters**: 204.8M
- **Trainable Parameters**: 118.8M
- **Frozen Parameters**: 86.0M (generator)

### Reconstruction Quality

- **MSE (Mean Squared Error)**: ~0.01-0.02
- **PSNR (Peak Signal-to-Noise Ratio)**: ~18-20 dB
- **Compression Ratio**: ~3072x (from 786,432 pixels to 64 tokens)

## 🔧 Troubleshooting

### Issue: "CUDA is not available"

**Solution**: Use CPU mode
```bash
python finetune_titok_mri.py --device cpu
```

### Issue: "Dataset not found"

**Solution**: Verify ACDC dataset path
```bash
ls /root/Documents/ICLR-Med/MedCompression/dataloader/acdc_img_datasets/
```

### Issue: "Out of memory"

**Solution**: Reduce batch size
```bash
python finetune_titok_mri.py --batch_size 2 --num_epochs 5
```

### Issue: "Model download fails"

**Solution**: Use local checkpoints
```bash
python finetune_titok_mri.py \
    --tokenizer_path ./checkpoints/tokenizer_titok_b64 \
    --generator_path ./checkpoints/generator_titok_b64
```

## 📈 Monitoring Training

Training logs are saved to the output directory:

```
outputs/
├── checkpoints/
│   ├── checkpoint_epoch_005.pt
│   ├── checkpoint_epoch_010.pt
│   └── best_model.pt
├── train_log.txt
└── config.yaml
```

Each checkpoint contains:
- Model state dictionary
- Optimizer state
- Training metrics
- Epoch information

## 🔬 Advanced Usage

### Custom Data Augmentation

```python
from med_mri.acdc_dataset import ACDCMRIDataset

dataset = ACDCMRIDataset(
    data_root="/path/to/data",
    split='train',
    image_size=(256, 256),
    augment=True,  # Enable augmentation
    max_images=100  # Limit for testing
)
```

### Inference

```python
model.eval()
with torch.no_grad():
    image = torch.randn(1, 3, 256, 256)

    # Encode to tokens
    tokens, _ = model.tokenizer.encode(image)

    # Decode to reconstruct
    reconstructed = model.tokenizer.decode(tokens)
```

### Evaluate on Test Set

```python
model.eval()
evaluator = TiTokMRIEvaluator()

test_metrics = validate(
    model, test_loader, device,
    epoch=-1, evaluator=evaluator
)
```

## 📚 References

### Papers

- **TiTok**: "An Image is Worth 32 Tokens for Reconstruction and Generation"
- **ACDC**: "Deep Learning Techniques for Medical Image Segmentation"

### Model Sources

- TiTok B-64: `yucornetto/tokenizer_titok_b64_imagenet`
- Generator: `yucornetto/generator_titok_b64_imagenet`

## 📝 Citation

If you use this module in your research, please cite:

```bibtex
@article{yu2024an,
  title={An Image is Worth 32 Tokens for Reconstruction and Generation},
  author={Yu, Qihang and Weber, Mark and Deng, Xueqing and others},
  journal={NeurIPS},
  year={2024}
}

@article{bernard2018deep,
  title={Deep Learning Techniques for Medical Image Segmentation},
  author={Bernard, Olivier and Lalande, Alain and others},
  journal={IEEE Transactions on Medical Imaging},
  year={2018}
}
```

## 🤝 Contributing

To improve this module:

1. Create a new branch: `git checkout -b feature/your-feature`
2. Make changes and test: `python test_finetune.py`
3. Commit: `git commit -am "Add feature"`
4. Push: `git push origin feature/your-feature`
5. Create a Pull Request

## 📧 Support

For issues or questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review test outputs: `python test_finetune.py`
3. Check model checkpoints directory for training logs
4. Verify dataset paths are correct

## 📋 Changelog

### v1.0 (2025-10-19)
- Initial release
- Full training pipeline
- ACDC dataset support
- TiTok model integration
- CPU and GPU support

## 📄 License

This module is part of the ICLR-Med project. See LICENSE file for details.

---

**Status**: ✅ Ready for use
**Last Updated**: October 19, 2025
**Tested On**: Python 3.13, PyTorch 2.9.0, ACDC Dataset
