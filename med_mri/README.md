# 🫀 TiTok MRI Fine-tuning

Fine-tuning TiTok model on ACDC heart MRI dataset for medical image compression.

**Features**: ~3000x compression ratio, CPU/GPU support, PyTorch implementation.

## 📊 Dataset Setup

### Convert ACDC Dataset (NIfTI → PNG)

If you have the original ACDC dataset in NIfTI format, convert it to PNG images:

```bash
# Navigate to med_mri directory
cd med_mri

# Convert NIfTI files to PNG (requires nibabel)
python convert_acdc.py

# Expected output structure:
# acdc_img_datasets/
# ├── patient001/
# │   ├── frame_000_slice_000.png
# │   ├── frame_000_slice_001.png
# │   └── ...
# ├── patient002/
# │   └── ...
# └── ...
```

**Requirements**: Install nibabel for NIfTI file handling
```bash
pip install nibabel
```

**Source**: Place original ACDC dataset at `./acdc_dataset/` (relative to med_mri directory)

### Pre-converted Dataset

Ensure ACDC dataset is available at:
```
../acdc_img_datasets/
```

## 🚀 Commands

### Install Dependencies
```bash
cd /userhome/cs3/fung0311/CVPR-2025/MedCompression
pip install torch torchvision transformers pillow numpy matplotlib tqdm pyyaml omegaconf
```

### Download Checkpoints
```bash
# Interactive model selection (recommended)
python download_checkpoints.py

# Download best performing model (non-interactive)
python download_checkpoints.py --non-interactive

# Download to custom directory
python download_checkpoints.py --output_dir ../my_checkpoints --non-interactive

# Download specific models
python download_checkpoints.py --models tokenizer_bl128_vae  # Best FID (0.84)
python download_checkpoints.py --models tokenizer_b64        # Balanced performance
python download_checkpoints.py --models tokenizer_bl128_vq   # VQ alternative

# List all available models
python download_checkpoints.py --list

**⚠️ Authentication Required:**
```bash
# These TiTok models require HuggingFace authentication
huggingface-cli login

# Or set environment variable:
# export HF_TOKEN=your_token_here
```

**Manual Download:** If automatic download fails, manually download from:
- https://huggingface.co/yucornetto/tokenizer_titok_bl128_vae_c16_imagenet
- https://huggingface.co/yucornetto/tokenizer_titok_b64_imagenet
```

**Model Performance (FID scores - lower is better):**
- `tokenizer_bl128_vae` - **FID: 0.84** ⭐ Best performance (default for H800)
- `tokenizer_sl256_vq` - FID: 1.03
- `tokenizer_bl128_vq` - FID: 1.49
- `tokenizer_b64` - FID: 1.70 (balanced size/performance)
- `tokenizer_bl64_vae` - FID: 1.25
- `tokenizer_ll32_vae` - FID: 1.61

**Available Models:**
- **VAE models**: Best reconstruction quality, suitable for H800 fine-tuning
- **VQ models**: Good balance of quality and speed
- **Standard models**: Smaller size, faster training

### Test Setup
```bash
cd med_mri
python test_finetune.py
```

### Train Model (CUDA - Recommended)
```bash
# Basic CUDA training
python finetune_titok_mri.py --batch_size 8 --num_epochs 20

# CUDA training with image saving
python finetune_titok_mri.py --batch_size 8 --num_epochs 20 --save_images --save_image_every 5
```

### Train Model (GPU via SLURM)
```bash
./gpu_run.sh train
```

### Train Model (CPU - Slow)
```bash
python finetune_titok_mri.py --batch_size 2 --num_epochs 5 --device cpu
```

### GPU Interactive Session
```bash
./gpu_run.sh bash
```

## ⚙️ Configuration

Key parameters in `finetune_titok_mri.py`:
- `--batch_size`: Training batch size (default: 8)
- `--num_epochs`: Number of epochs (default: 20)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--device`: Device to use (`cuda`/`cpu`/auto, default: `cuda`)
- `--save_images`: Save validation/test image samples (flag)
- `--save_image_every`: Save validation images every N epochs (default: 5)

## 🔧 Basic Troubleshooting

**CUDA not available**: Use `--device cpu`
```bash
python finetune_titok_mri.py --device cpu
```

**Out of memory**: Reduce batch size
```bash
python finetune_titok_mri.py --batch_size 2
```

**Dataset not found**: Check path
```bash
ls ../acdc_img_datasets/
```
