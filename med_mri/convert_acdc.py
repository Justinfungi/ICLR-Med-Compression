#!/usr/bin/env python3
"""
🫀 ACDC Dataset Converter - Unified Script

Converts ACDC cardiac MRI dataset from NIfTI format to PNG images

Source:  /root/Documents/ICLR-Med/MedCompression/acdc_dataset/
Output:  /root/Documents/ICLR-Med/MedCompression/acdc_img_datasets/

Usage:
    python3 convert_acdc.py
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ACDCConverter:
    """Convert ACDC dataset from NIfTI to PNG"""

    def __init__(self, source_dir, output_dir):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.total_images = 0

    def load_nifti(self, filepath):
        """Load NIfTI file using nibabel or scipy"""
        try:
            import nibabel as nib
            nib_img = nib.load(filepath)
            return nib_img.get_fdata()
        except ImportError:
            logger.warning("nibabel not available, trying scipy...")
            try:
                from scipy.io import savemat
                import gzip
                # Fallback: try to read as generic binary
                logger.error("scipy fallback not implemented. Please install nibabel: pip install nibabel")
                raise ImportError("nibabel required")
            except Exception as e:
                logger.error(f"Cannot load {filepath}: {e}")
                raise

    def normalize_image(self, img):
        """Normalize image to 0-255 range"""
        img = np.asarray(img, dtype=np.float32)

        if img.max() > 0:
            img = (img - img.min()) / (img.max() - img.min() + 1e-6)
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        return img

    def save_image(self, data, output_path):
        """Save image as PNG"""
        normalized = self.normalize_image(data)
        pil_img = Image.fromarray(normalized)
        pil_img.save(output_path)
        self.total_images += 1

    def process_patient(self, patient_dir):
        """Process single patient directory"""
        patient_name = patient_dir.name
        output_patient_dir = self.output_dir / patient_name
        output_patient_dir.mkdir(parents=True, exist_ok=True)

        # Find NIfTI files
        nifti_files = sorted(patient_dir.glob('*.nii*'))

        for nifti_file in nifti_files:
            try:
                # Load NIfTI
                img_data = self.load_nifti(nifti_file)
                img_data = np.asarray(img_data)

                # Handle 3D images (multiple slices)
                if len(img_data.shape) == 3:
                    for slice_idx in range(img_data.shape[2]):
                        slice_data = img_data[:, :, slice_idx]
                        stem = nifti_file.stem.replace('.nii', '')
                        output_path = output_patient_dir / f"{stem}_slice_{slice_idx:03d}.png"
                        self.save_image(slice_data, output_path)

                # Handle 2D images
                elif len(img_data.shape) == 2:
                    output_path = output_patient_dir / f"{nifti_file.stem}.png"
                    self.save_image(img_data, output_path)

            except Exception as e:
                logger.warning(f"Error processing {nifti_file.name}: {e}")
                continue

    def convert(self):
        """Convert entire dataset"""
        # Find patient directories
        patient_dirs = sorted([d for d in self.source_dir.glob('patient*')])

        if not patient_dirs:
            logger.error(f"No patient directories found in {self.source_dir}")
            return False

        logger.info(f"Found {len(patient_dirs)} patient directories")

        # Process each patient
        for patient_dir in tqdm(patient_dirs, desc="Converting patients"):
            self.process_patient(patient_dir)

        return True


def main():
    """Main function"""
    source_dir = "/root/Documents/ICLR-Med/MedCompression/acdc_dataset"
    output_dir = "/root/Documents/ICLR-Med/MedCompression/acdc_img_datasets"

    print("╔" + "=" * 70 + "╗")
    print("║" + "  🫀 ACDC Dataset Converter (NIfTI → PNG)".center(70) + "║")
    print("╚" + "=" * 70 + "╝")
    print()

    # Verify source exists
    if not Path(source_dir).exists():
        logger.error(f"❌ Source directory not found: {source_dir}")
        return False

    logger.info(f"Source: {source_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info("")

    # Check dependencies
    try:
        import nibabel
        logger.info("✅ nibabel available")
    except ImportError:
        logger.error("❌ nibabel not installed. Install with: pip install nibabel")
        return False

    # Create converter
    converter = ACDCConverter(source_dir, output_dir)

    # Convert dataset
    logger.info("Starting conversion...")
    print()
    success = converter.convert()

    if not success:
        logger.error("❌ Conversion failed!")
        return False

    print()
    print("=" * 72)
    logger.info(f"✅ Conversion completed successfully!")
    logger.info(f"   Total images converted: {converter.total_images}")
    logger.info(f"   Output directory: {output_dir}")
    print("=" * 72)

    # Show statistics
    output_path = Path(output_dir)
    patient_dirs = list(output_path.glob('patient*'))
    total_images = sum(len(list(d.glob('*.png'))) for d in patient_dirs)

    logger.info(f"\n📊 Statistics:")
    logger.info(f"   Patient directories: {len(patient_dirs)}")
    logger.info(f"   Total PNG images:    {total_images}")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n⚠️  Conversion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
