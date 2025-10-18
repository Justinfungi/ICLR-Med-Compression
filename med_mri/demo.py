#!/usr/bin/env python3
"""
TiTok Tokenizer Demo - Python Script Version

This script demonstrates the TiTok tokenizer functionality:
1. Tokenize images into 32 discrete tokens
2. Reconstruct images from tokens
3. Generate new images from tokens
"""

import demo_util
import numpy as np
import torch
from PIL import Image
import imagenet_classes
import os
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
    """Load the TiTok tokenizer and generator models"""
    print("🔄 Loading TiTok models...")

    # Load configuration
    config = demo_util.get_config("configs/infer/TiTok/titok_l32.yaml")
    print(f"📋 Configuration loaded: {config}")

    # Load tokenizer
    # supported tokenizer: [tokenizer_titok_l32_imagenet, tokenizer_titok_b64_imagenet, tokenizer_titok_s128_imagenet]
    titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)
    print("✅ Tokenizer loaded successfully")

    # Load generator
    # supported generator: [generator_titok_l32_imagenet, generator_titok_b64_imagenet, generator_titok_s128_imagenet]
    titok_generator = ImageBert.from_pretrained("yucornetto/generator_titok_l32_imagenet")
    titok_generator.eval()
    titok_generator.requires_grad_(False)
    print("✅ Generator loaded successfully")

    # Alternative loading method (commented out):
    # hf_hub_download(repo_id="fun-research/TiTok", filename="tokenizer_titok_l32.bin", local_dir="./")
    # hf_hub_download(repo_id="fun-research/TiTok", filename="generator_titok_l32.bin", local_dir="./")

    return titok_tokenizer, titok_generator


def setup_device(tokenizer, generator):
    """Setup device (CUDA) for models"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🎯 Using device: {device}")

    tokenizer = tokenizer.to(device)
    generator = generator.to(device)
    print("✅ Models moved to device")

    return tokenizer, generator, device


def tokenize_and_reconstruct(img_path, tokenizer, device, save_output=True):
    """Tokenize an image into 32 discrete tokens and reconstruct it"""
    print(f"🔄 Processing image: {img_path}")

    # Load and preprocess image
    original_image = Image.open(img_path)
    image = torch.from_numpy(np.array(original_image).astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0

    # Tokenize
    encoded_tokens = tokenizer.encode(image.to(device))[1]["min_encoding_indices"]

    # Reconstruct
    reconstructed_image = tokenizer.decode_tokens(encoded_tokens)
    reconstructed_image = torch.clamp(reconstructed_image, 0.0, 1.0)
    reconstructed_image = (reconstructed_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]

    # Convert to PIL Image
    reconstructed_image = Image.fromarray(reconstructed_image)

    print(f"📊 Input Image is represented by codes {encoded_tokens} with shape {encoded_tokens.shape}")

    if save_output:
        # Save images for comparison
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        original_image.save(f"{output_dir}/{base_name}_original.png")
        reconstructed_image.save(f"{output_dir}/{base_name}_reconstructed.png")

        print(f"💾 Original image saved to: {output_dir}/{base_name}_original.png")
        print(f"💾 Reconstructed image saved to: {output_dir}/{base_name}_reconstructed.png")

        # Display using matplotlib (optional)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(original_image)
        ax1.set_title("Original Image")
        ax1.axis('off')

        ax2.imshow(reconstructed_image)
        ax2.set_title("Reconstructed Image")
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{base_name}_comparison.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"💾 Comparison image saved to: {output_dir}/{base_name}_comparison.png")

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
        "assets/ILSVRC2012_val_00008636.png",
        "assets/ILSVRC2012_val_00010240.png"
    ]

    for img_path in test_images:
        if os.path.exists(img_path):
            tokenize_and_reconstruct(img_path, titok_tokenizer, device)
            print("-" * 30)
        else:
            print(f"⚠️  Warning: Image not found: {img_path}")

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

    print("\n" + "=" * 50)
    print("✅ Demo completed successfully!")
    print("📁 Check the 'output' directory for generated images and reconstructions.")


if __name__ == "__main__":
    main()
