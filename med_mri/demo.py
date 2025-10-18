#!/usr/bin/env python3
"""
TiTok Tokenizer Demo - Python Script Version

This script demonstrates the TiTok tokenizer functionality:
1. Tokenize images into 32 discrete tokens
2. Reconstruct images from tokens
3. Generate new images from tokens
"""

import sys
import os
sys.path.append('../1d-tokenizer')
import demo_util
import numpy as np
import torch
from PIL import Image
import imagenet_classes
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from modeling.maskgit import ImageBert
from modeling.titok import TiTok


def setup_environment():
    """Setup PyTorch environment and random seed"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    print("✅ Environment setup complete")


def load_models():
    """Load the TiTok tokenizer model (generator disabled for compatibility)"""
    print("🔄 Loading TiTok tokenizer...")

    # Load tokenizer
    # Option 1: Use local tokenizer (recommended for your custom model)
    local_tokenizer_path = "checkpoints/tokenizer_titok_bl128_vae_c16_imagenet"
    if os.path.exists(local_tokenizer_path):
        print(f"✅ 使用本地tokenizer: {local_tokenizer_path}")
        titok_tokenizer = TiTok.from_pretrained(local_tokenizer_path)
    else:
        # Option 2: Fallback to HuggingFace tokenizer
        # supported tokenizer: [tokenizer_titok_l32_imagenet, tokenizer_titok_b64_imagenet, tokenizer_titok_s128_imagenet]
        print("⚠️ 本地tokenizer未找到，使用HuggingFace tokenizer")
        titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)
    print("✅ Tokenizer loaded successfully")

    # Skip generator loading for now due to compatibility issues
    print("⚠️ Generator暂时禁用 - 专注于tokenization和reconstruction功能")
    titok_generator = None

    return titok_tokenizer, titok_generator


def setup_device(tokenizer, generator):
    """Setup device (CUDA) for models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    tokenizer = tokenizer.to(device)
    if generator is not None:
        generator = generator.to(device)
    print("✅ Models moved to device")

    return tokenizer, generator, device


def tokenize_and_reconstruct(img_path, tokenizer, device, save_output=True):
    """Tokenize an image into discrete tokens and reconstruct it"""
    print(f"🔄 Processing image: {img_path}")

    # Load and preprocess image
    original_image = Image.open(img_path)
    
    # Handle grayscale images (convert to RGB for consistency)
    if original_image.mode == 'L':
        print("📝 Converting grayscale image to RGB")
        original_image = original_image.convert('RGB')
    
    # Convert to tensor and resize to expected size (256x256)
    # The model expects 256x256 images based on the config
    original_image = original_image.resize((256, 256), Image.LANCZOS)
    image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
    print(f"📊 Input image tensor shape: {image.shape}")

    # Tokenize - handle both VQ and VAE modes
    with torch.no_grad():
        encode_result = tokenizer.encode(image.to(device))
        
        # Check if we have a tuple (tokens, info_dict) or just tokens
        if isinstance(encode_result, tuple) and len(encode_result) == 2:
            tokens, info_dict = encode_result
            
            # Check quantization mode
            if hasattr(tokenizer, 'quantize_mode'):
                if tokenizer.quantize_mode == "vq":
                    # VQ mode: use min_encoding_indices
                    encoded_tokens = info_dict["min_encoding_indices"]
                elif tokenizer.quantize_mode == "vae":
                    # VAE mode: sample from posterior distribution
                    posteriors = info_dict
                    encoded_tokens = posteriors.sample()
                else:
                    # Fallback: try to get tokens directly
                    encoded_tokens = tokens
            else:
                # Fallback: try min_encoding_indices first, then sample
                if "min_encoding_indices" in info_dict:
                    encoded_tokens = info_dict["min_encoding_indices"]
                elif hasattr(info_dict, 'sample'):
                    encoded_tokens = info_dict.sample()
                else:
                    encoded_tokens = tokens
        else:
            # Direct token return
            encoded_tokens = encode_result

    # Reconstruct
    with torch.no_grad():
        if hasattr(tokenizer, 'decode_tokens'):
            reconstructed_image = tokenizer.decode_tokens(encoded_tokens)
        else:
            # Fallback to decode method
            reconstructed_image = tokenizer.decode(encoded_tokens)
        reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
        reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]

    # Convert to PIL Image
    reconstructed_image = Image.fromarray(reconstructed_image)

    print(f"📊 Input Image is represented by tokens with shape {encoded_tokens.shape}")
    if hasattr(encoded_tokens, 'min') and hasattr(encoded_tokens, 'max'):
        print(f"🔢 Token values range: [{encoded_tokens.min().item()}, {encoded_tokens.max().item()}]")
    else:
        print(f"🔢 Token type: {type(encoded_tokens)}")

    if save_output:
        # Save images for comparison
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = "outputs/demos"
        os.makedirs(output_dir, exist_ok=True)

        original_image.save(f"{output_dir}/{base_name}_original.png")
        reconstructed_image.save(f"{output_dir}/{base_name}_reconstructed.png")

        original_path = f"{output_dir}/{base_name}_original.png"
        reconstructed_path = f"{output_dir}/{base_name}_reconstructed.png"
        
        print(f"💾 Original image saved to: {os.path.abspath(original_path)}")
        print(f"💾 Reconstructed image saved to: {os.path.abspath(reconstructed_path)}")

        # Display using matplotlib (optional)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.imshow(original_image)
        ax1.set_title("Original MRI Image")
        ax1.axis('off')

        ax2.imshow(reconstructed_image)
        ax2.set_title("Reconstructed Image")
        ax2.axis('off')

        plt.tight_layout()
        comparison_path = f"{output_dir}/{base_name}_comparison.png"
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Comparison image saved to: {os.path.abspath(comparison_path)}")

    return encoded_tokens, reconstructed_image


def generate_image_from_tokens(generator, tokenizer, labels, device, guidance_scale=3.5, randomize_temperature=1.0, num_sample_steps=8, save_output=True):
    """Generate images from discrete tokens"""
    print(f"🎨 Generating images for labels: {labels}")

    # Generate images
    generated_images = demo_util.sample_fn(
        generator=generator,
        tokenizer=tokenizer,
        labels=labels,
        guidance_scale=guidance_scale,
        randomize_temperature=randomize_temperature,
        num_sample_steps=num_sample_steps,
        device=device
    )

    if save_output:
        output_dir = "output/generated"
        os.makedirs(output_dir, exist_ok=True)

        for i in range(generated_images.shape[0]):
            label = labels[i]
            class_name = imagenet_classes.imagenet_idx2classname[label]
            print(f"🏷️  Label {label}: {class_name}")

            # Convert to PIL Image
            img_array = generated_images[i]
            generated_image = Image.fromarray(img_array)

            # Save generated image
            filename = f"{output_dir}/generated_{label}_{class_name.replace(' ', '_')}.png"
            generated_image.save(filename)
            print(f"💾 Generated image saved to: {filename}")

            # Optional: display using matplotlib
            plt.figure(figsize=(5, 5))
            plt.imshow(generated_image)
            plt.title(f"Generated: {class_name}")
            plt.axis('off')
            plt.savefig(filename.replace('.png', '_display.png'), dpi=150, bbox_inches='tight')
            plt.close()

    return generated_images


def main():
    """Main demo function"""
    print("🚀 Starting TiTok Tokenizer Demo")
    print("=" * 50)

    # Setup
    setup_environment()

    # Load models
    titok_tokenizer, titok_generator = load_models()

    # Setup device
    titok_tokenizer, titok_generator, device = setup_device(titok_tokenizer, titok_generator)

    print("\n" + "=" * 50)
    print("🖼️  TOKENIZATION AND RECONSTRUCTION DEMO")
    print("=" * 50)

    # Test images
    test_images = [
        "../acdc_img_datasets/patient001/patient001_frame01_slice_000.png"
    ]

    for img_path in test_images:
        if os.path.exists(img_path):
            tokenize_and_reconstruct(img_path, titok_tokenizer, device)
            print("-" * 30)
        else:
            print(f"⚠️  Warning: Image not found: {img_path}")

    # Skip generation demo since generator is disabled
    if titok_generator is not None:
        print("\n" + "=" * 50)
        print("🎨 IMAGE GENERATION DEMO")
        print("=" * 50)

        # Generate sample images
        sample_labels = [torch.randint(0, 999, size=(1,)).item() for _ in range(3)]  # Generate 3 random samples
        print(f"🎲 Random labels generated: {sample_labels}")

        generated_images = generate_image_from_tokens(
            titok_generator,
            titok_tokenizer,
            sample_labels,
            device,
            guidance_scale=3.5,
            randomize_temperature=1.0,
            num_sample_steps=8
        )
    else:
        print("\n" + "=" * 50)
        print("⚠️ IMAGE GENERATION DEMO SKIPPED")
        print("Generator未加载 - 专注于tokenization和reconstruction功能")
        print("=" * 50)

    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    output_abs_path = os.path.abspath("outputs/demos")
    print(f"📁 Check the output directory for generated images and reconstructions:")
    print(f"   {output_abs_path}")


if __name__ == "__main__":
    main()
