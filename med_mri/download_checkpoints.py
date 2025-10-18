#!/usr/bin/env python3
"""
Download TiTok Model Checkpoints

Downloads TiTok tokenizer and generator checkpoints for MRI fine-tuning.

Usage:
    python download_checkpoints.py
    python download_checkpoints.py --output_dir ./my_checkpoints
"""

import os
import sys
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# TiTok model configurations
TITOK_MODELS = {
    # Standard models
    'tokenizer_b64': {
        'repo_id': 'yucornetto/tokenizer_titok_b64_imagenet',
        'filename': 'tokenizer_titok_b64',
        'description': 'TiTok B-64 Tokenizer'
    },
    'tokenizer_l32': {
        'repo_id': 'yucornetto/tokenizer_titok_l32_imagenet',
        'filename': 'tokenizer_titok_l32',
        'description': 'TiTok L-32 Tokenizer'
    },
    'tokenizer_s128': {
        'repo_id': 'yucornetto/tokenizer_titok_s128_imagenet',
        'filename': 'tokenizer_titok_s128',
        'description': 'TiTok S-128 Tokenizer'
    },
    'generator_b64': {
        'repo_id': 'yucornetto/generator_titok_b64_imagenet',
        'filename': 'generator_titok_b64',
        'description': 'TiTok B-64 Generator'
    },
    'generator_l32': {
        'repo_id': 'yucornetto/generator_titok_l32_imagenet',
        'filename': 'generator_titok_l32',
        'description': 'TiTok L-32 Generator'
    },
    'generator_s128': {
        'repo_id': 'yucornetto/generator_titok_s128_imagenet',
        'filename': 'generator_titok_s128',
        'description': 'TiTok S-128 Generator'
    },
    # VQ models
    'tokenizer_bl64_vq': {
        'repo_id': 'yucornetto/tokenizer_titok_bl64_vq8k_imagenet',
        'filename': 'tokenizer_titok_bl64_vq8k',
        'description': 'TiTok BL-64 VQ Tokenizer'
    },
    'tokenizer_bl128_vq': {
        'repo_id': 'yucornetto/tokenizer_titok_bl128_vq8k_imagenet',
        'filename': 'tokenizer_titok_bl128_vq8k',
        'description': 'TiTok BL-128 VQ Tokenizer'
    },
    'tokenizer_sl256_vq': {
        'repo_id': 'yucornetto/tokenizer_titok_sl256_vq8k_imagenet',
        'filename': 'tokenizer_titok_sl256_vq8k',
        'description': 'TiTok SL-256 VQ Tokenizer'
    },
    # VAE models
    'tokenizer_ll32_vae': {
        'repo_id': 'yucornetto/tokenizer_titok_ll32_vae_c16_imagenet',
        'filename': 'tokenizer_titok_ll32_vae_c16',
        'description': 'TiTok LL-32 VAE Tokenizer'
    },
    'tokenizer_bl64_vae': {
        'repo_id': 'yucornetto/tokenizer_titok_bl64_vae_c16_imagenet',
        'filename': 'tokenizer_titok_bl64_vae_c16',
        'description': 'TiTok BL-64 VAE Tokenizer'
    },
    'tokenizer_bl128_vae': {
        'repo_id': 'yucornetto/tokenizer_titok_bl128_vae_c16_imagenet',
        'filename': 'tokenizer_titok_bl128_vae_c16',
        'description': 'TiTok BL-128 VAE Tokenizer'
    }
}


def download_checkpoint(repo_id, output_dir, filename=None):
    """
    Download a single checkpoint from HuggingFace Hub

    Args:
        repo_id: HuggingFace repository ID
        output_dir: Output directory path
        filename: Optional specific filename to save as

    Returns:
        Path to downloaded checkpoint directory
    """
    try:
        logger.info(f"Downloading {repo_id}...")

        # Download the entire repository
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=output_dir / repo_id.split('/')[-1],
            local_dir_use_symlinks=False
        )

        logger.info(f"✅ Downloaded {repo_id} to {local_dir}")
        return local_dir

    except Exception as e:
        logger.error(f"❌ Failed to download {repo_id}: {e}")
        return None


def check_huggingface_access():
    """Check if HuggingFace Hub is accessible"""
    try:
        api = HfApi()
        # Try to list models (this will fail if no access)
        api.list_models(limit=1)
        return True
    except Exception as e:
        logger.error(f"❌ HuggingFace Hub access failed: {e}")
        logger.error("These TiTok models require authentication to download.")
        logger.info("🔐 Please login to HuggingFace first:")
        logger.info("   huggingface-cli login")
        logger.info("   # Or set HF_TOKEN environment variable")
        logger.info("")
        logger.info("Alternatively, you can manually download the models:")
        logger.info("   https://huggingface.co/yucornetto/tokenizer_titok_bl128_vae_c16_imagenet")
        logger.info("   https://huggingface.co/yucornetto/tokenizer_titok_b64_imagenet")
        return False


def display_model_menu():
    """Display interactive model selection menu"""
    print("╔" + "=" * 80 + "╗")
    print("║" + "  🫀 TiTok Model Selection Menu".center(80) + "║")
    print("╚" + "=" * 80 + "╝")
    print()

    # Group models by type
    standard_models = []
    vq_models = []
    vae_models = []

    for key, model in TITOK_MODELS.items():
        if 'vq' in key:
            vq_models.append((key, model))
        elif 'vae' in key:
            vae_models.append((key, model))
        else:
            standard_models.append((key, model))

    print("Standard Models:")
    for i, (key, model) in enumerate(standard_models, 1):
        recommended = " (recommended)" if key == 'tokenizer_bl128_vae' else ""
        print(f"  {i:2d}. {key:20} - {model['description']}{recommended}")

    print("\nVQ Models (Vector Quantization):")
    for i, (key, model) in enumerate(vq_models, len(standard_models) + 1):
        print(f"  {i:2d}. {key:20} - {model['description']}")

    print("\nVAE Models (Variational Autoencoder) - Best for medical imaging:")
    # Sort VAE models to put recommended first
    vae_models_sorted = sorted(vae_models, key=lambda x: 0 if x[0] == 'tokenizer_bl128_vae' else 1)
    for i, (key, model) in enumerate(vae_models_sorted, len(standard_models) + len(vq_models) + 1):
        recommended = " ⭐ BEST PERFORMANCE" if key == 'tokenizer_bl128_vae' else ""
        print(f"  {i:2d}. {key:20} - {model['description']}{recommended}")

    print()
    return standard_models + vq_models + vae_models_sorted


def interactive_model_selection():
    """Interactive model selection"""
    all_models = display_model_menu()
    total_models = len(all_models)

    while True:
        try:
            choice = input(f"Select model to download (1-{total_models}) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                print("Download cancelled.")
                return None

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < total_models:
                selected_key, selected_model = all_models[choice_idx]
                print(f"\nSelected: {selected_model['description']}")
                print(f"Model key: {selected_key}")
                return selected_key
            else:
                print(f"Invalid choice. Please enter a number between 1 and {total_models}.")

        except ValueError:
            print("Invalid input. Please enter a number or 'q' to quit.")
        except KeyboardInterrupt:
            print("\nDownload cancelled.")
            return None


def main():
    parser = argparse.ArgumentParser(description='Download TiTok Model Checkpoints')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                      help='Output directory for checkpoints (default: ./checkpoints)')
    parser.add_argument('--models', nargs='+', choices=list(TITOK_MODELS.keys()),
                      help=f'Models to download. Available: {list(TITOK_MODELS.keys())}')
    parser.add_argument('--list', action='store_true',
                      help='List available models and exit')
    parser.add_argument('--non-interactive', action='store_true',
                      help='Skip interactive selection and use defaults')

    args = parser.parse_args()

    # List available models if requested
    if args.list:
        display_model_menu()
        print("\nUsage examples:")
        print("  python download_checkpoints.py --models tokenizer_b64")
        print("  python download_checkpoints.py --models tokenizer_b64 generator_b64")
        print("  python download_checkpoints.py --models tokenizer_bl128_vq")
        print("  python download_checkpoints.py  # Interactive selection")
        return True

    print("╔" + "=" * 60 + "╗")
    print("║" + "  🫀 TiTok Checkpoint Downloader".center(60) + "║")
    print("╚" + "=" * 60 + "╝")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")

    # Check HuggingFace access
    if not check_huggingface_access():
        return False

    # Determine which models to download
    if args.models:
        models_to_download = args.models
        logger.info(f"📋 Downloading specified models: {', '.join(models_to_download)}")
    elif args.non_interactive:
        # Non-interactive mode: use default
        models_to_download = ['tokenizer_bl128_vae']  # Best FID (0.84), suitable for H800
        logger.info("🎯 Non-interactive mode: downloading best performing model for H800 fine-tuning: tokenizer_bl128_vae (FID: 0.84)")
    else:
        # Interactive mode: let user choose
        logger.info("🔍 Starting interactive model selection...")
        selected_model = interactive_model_selection()
        if selected_model is None:
            return False  # User cancelled
        models_to_download = [selected_model]

    downloaded_paths = []

    # Download selected models
    for model_key in models_to_download:
        model_config = TITOK_MODELS[model_key]
        logger.info(f"📥 {model_config['description']}")

        checkpoint_path = download_checkpoint(
            model_config['repo_id'],
            output_dir,
            model_config['filename']
        )

        if checkpoint_path:
            downloaded_paths.append(checkpoint_path)

    # Summary
    if downloaded_paths:
        print()
        print("=" * 65)
        logger.info("✅ Download completed successfully!")
        logger.info(f"   Checkpoints saved to: {output_dir.absolute()}")
        print()

        # List downloaded models
        logger.info("📁 Downloaded checkpoints:")
        for path in downloaded_paths:
            model_name = Path(path).name
            logger.info(f"   • {model_name}: {path}")

        print()
        logger.info("💡 Usage in finetune_titok_mri.py:")
        logger.info("   --tokenizer_path ./checkpoints/tokenizer_titok_b64_imagenet")
        logger.info("   --generator_path ./checkpoints/generator_titok_b64_imagenet")

        return True
    else:
        logger.error("❌ No checkpoints were downloaded successfully.")
        return False


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
