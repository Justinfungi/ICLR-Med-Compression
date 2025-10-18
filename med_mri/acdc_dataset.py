#!/usr/bin/env python3
"""
ACDC MRI Dataset Loader

Loads and processes ACDC heart MRI dataset for TiTok fine-tuning
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, Tuple, Optional, List
from collections import defaultdict

logger = logging.getLogger(__name__)


class ACDCMRIDataset(Dataset):
    """ACDC MRI Dataset for image reconstruction and compression tasks"""

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (256, 256),
        augment: bool = False,
        max_images: Optional[int] = None,
    ):
        """
        Initialize ACDC MRI Dataset

        Args:
            data_root: Root directory containing ACDC dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (H, W)
            augment: Enable data augmentation
            max_images: Maximum number of images to load (for testing)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.max_images = max_images

        # Define patient splits (70 train, 20 val, 10 test)
        all_patients = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        n_patients = len(all_patients)

        split_indices = {
            'train': list(range(0, int(0.7 * n_patients))),
            'val': list(range(int(0.7 * n_patients), int(0.9 * n_patients))),
            'test': list(range(int(0.9 * n_patients), n_patients))
        }

        self.patients = [all_patients[i] for i in split_indices[split]]

        # Collect all images
        self.images = []
        for patient_dir in self.patients:
            png_files = sorted(patient_dir.glob('*.png'))
            for png_file in png_files:
                self.images.append((png_file, patient_dir.name))

        if self.max_images:
            self.images = self.images[:self.max_images]

        logger.info(f"✅ ACDC MRI数据集加载完成: {len(self.images)} 张图像")

        # Define augmentation transforms
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ])
        else:
            self.transform = None

        # Resize transform
        self.resize = transforms.Resize(self.image_size, interpolation=Image.BILINEAR)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path, patient_id = self.images[idx]

        # Load image
        img = Image.open(image_path).convert('RGB')

        # Resize
        img = self.resize(img)

        # Apply augmentation
        if self.augment and self.transform:
            img = self.transform(img)

        # Convert to tensor and normalize
        img_tensor = transforms.ToTensor()(img)

        return {
            'image': img_tensor,
            'patient_id': patient_id,
            'filename': image_path.stem
        }


def create_data_loaders(
    data_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    augment: bool = True,
    max_images_per_split: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """
    Create data loaders for ACDC dataset

    Args:
        data_root: Root directory of ACDC dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        image_size: Target image size
        augment: Enable augmentation for training split
        max_images_per_split: Max images per split (for testing)

    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """

    # Print dataset statistics
    data_root_path = Path(data_root)

    if data_root_path.exists():
        patient_dirs = [d for d in data_root_path.iterdir() if d.is_dir()]
        total_images = sum(len(list(d.glob('*.png'))) for d in patient_dirs)
        print(f"📁 发现 {len(patient_dirs)} 个患者目录")
        print(f"🖼️ 总共 {total_images} 张PNG图像")

    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        dataset = ACDCMRIDataset(
            data_root=data_root,
            split=split,
            image_size=image_size,
            augment=augment if split == 'train' else False,
            max_images=max_images_per_split
        )
        datasets[split] = dataset

        # Print split info
        n_patients_in_split = len(dataset.patients)
        n_images_in_split = len(dataset)
        print(f"📊 {split.upper()}分割: {n_patients_in_split} 患者, {n_images_in_split} 图像")

    # Create data loaders
    data_loaders = {}
    for split in ['train', 'val', 'test']:
        data_loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers if split == 'train' else 0,
            pin_memory=True,
            drop_last=(split == 'train')
        )

        n_batches = len(data_loaders[split])
        print(f"{split.capitalize()}: {len(datasets[split])} 图像, {n_batches} batches")

    print("\n📊 数据加载器统计:")
    for split, loader in data_loaders.items():
        print(f"{split.capitalize()}: {len(loader.dataset)} 图像, {len(loader)} batches")

    return data_loaders


if __name__ == "__main__":
    # Test dataset loading
    data_root = "/root/Documents/ICLR-Med/MedCompression/dataloader/acdc_img_datasets"

    if os.path.exists(data_root):
        loaders = create_data_loaders(
            data_root=data_root,
            batch_size=4,
            num_workers=0,
            image_size=(256, 256),
            augment=False
        )

        # Get one batch
        batch = next(iter(loaders['train']))
        print(f"\nBatch shape: {batch['image'].shape}")
        print(f"Patient ID: {batch['patient_id']}")
    else:
        print(f"Dataset not found at: {data_root}")
