#!/usr/bin/env python3
"""
🫀 ACDC Dataset Converter - Unified Script

Converts ACDC cardiac MRI dataset from NIfTI format to PNG images

Source:  /root/Documents/ICLR-Med/MedCompression/acdc_dataset/
Output:  /root/Documents/ICLR-Med/MedCompression/acdc_img_datasets/

Usage:
    # Extract keyframes only (default, faster processing)
    python3 convert_acdc.py --extract_mode keyframe

    # Extract all time frames (complete cardiac cycle)
    python3 convert_acdc.py --extract_mode all_frames

    # Custom paths
    python3 convert_acdc.py --source_dir ./data/acdc --output_dir ./output/images
"""

import os
import sys
import argparse
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

    def __init__(self, source_dir, output_dir, extract_mode='keyframe'):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.extract_mode = extract_mode  # 'keyframe' or 'all_frames'
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

        # Find NIfTI files (exclude ground truth segmentation masks)
        nifti_files = sorted(patient_dir.glob('*.nii*'))

        # Filter out ground truth files (_gt suffix)
        nifti_files = [f for f in nifti_files if '_gt' not in f.name]

        if not nifti_files:
            logger.warning(f"No MRI image files found in {patient_dir} (ground truth files excluded)")
            return

        for nifti_file in nifti_files:
            try:
                # Load NIfTI
                img_data = self.load_nifti(nifti_file)
                img_data = np.asarray(img_data)

                # Handle 4D images (height, width, slices, time_frames)
                if len(img_data.shape) == 4:
                    if self.extract_mode == 'all_frames':
                        # Extract all time frames
                        for slice_idx in range(img_data.shape[2]):
                            for frame_idx in range(img_data.shape[3]):
                                slice_data = img_data[:, :, slice_idx, frame_idx]
                                stem = nifti_file.stem.replace('.nii', '')
                                output_path = output_patient_dir / f"{stem}_slice_{slice_idx:03d}_frame_{frame_idx:03d}.png"
                                self.save_image(slice_data, output_path)
                    elif self.extract_mode == 'keyframe':
                        # Extract only keyframe (typically first frame, or could be middle frame)
                        # For cardiac MRI, often the end-diastolic frame is most representative
                        keyframe_idx = img_data.shape[3] // 2  # Use middle frame as keyframe
                        for slice_idx in range(img_data.shape[2]):
                            slice_data = img_data[:, :, slice_idx, keyframe_idx]
                            stem = nifti_file.stem.replace('.nii', '')
                            output_path = output_patient_dir / f"{stem}_slice_{slice_idx:03d}_keyframe.png"
                            self.save_image(slice_data, output_path)

                # Handle 3D images (multiple slices, assuming single time frame)
                elif len(img_data.shape) == 3:
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
        # Find patient directories in both training and testing folders
        patient_dirs = []

        # Check for direct patient directories (legacy structure)
        direct_patients = sorted([d for d in self.source_dir.glob('patient*')])
        patient_dirs.extend(direct_patients)

        # Check training directory
        training_dir = self.source_dir / "training"
        if training_dir.exists():
            training_patients = sorted([d for d in training_dir.glob('patient*')])
            patient_dirs.extend(training_patients)
            logger.info(f"Found {len(training_patients)} patients in training/")

        # Check testing directory
        testing_dir = self.source_dir / "testing"
        if testing_dir.exists():
            testing_patients = sorted([d for d in testing_dir.glob('patient*')])
            patient_dirs.extend(testing_patients)
            logger.info(f"Found {len(testing_patients)} patients in testing/")

        if not patient_dirs:
            logger.error(f"No patient directories found in {self.source_dir} or its subdirectories")
            logger.error("Expected structure: acdc_dataset/{training,testing}/patientXXX/")
            return False

        logger.info(f"Total: Found {len(patient_dirs)} patient directories")

        # Process each patient
        for patient_dir in tqdm(patient_dirs, desc="Converting patients"):
            self.process_patient(patient_dir)

        return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="🫀 ACDC Dataset Converter - Convert NIfTI cardiac MRI to PNG images"
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="./acdc_dataset",
        help="Source directory containing ACDC dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./acdc_img_datasets",
        help="Output directory for PNG images"
    )
    parser.add_argument(
        "--extract_mode",
        type=str,
        choices=['keyframe', 'all_frames'],
        default='keyframe',
        help="Extraction mode: 'keyframe' (extract representative frame) or 'all_frames' (extract all time frames)"
    )

    args = parser.parse_args()

    print("╔" + "=" * 70 + "╗")
    print("║" + "  🫀 ACDC Dataset Converter (NIfTI → PNG)".center(70) + "║")
    print("╚" + "=" * 70 + "╝")
    print()

    # Verify source exists
    if not Path(args.source_dir).exists():
        logger.error(f"❌ Source directory not found: {args.source_dir}")
        return False

    logger.info(f"Source: {args.source_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Extract mode: {args.extract_mode}")
    logger.info("")

    # Check dependencies
    try:
        import nibabel
        logger.info("✅ nibabel available")
    except ImportError:
        logger.error("❌ nibabel not installed. Install with: pip install nibabel")
        return False

    # Create converter
    converter = ACDCConverter(args.source_dir, args.output_dir, args.extract_mode)

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
