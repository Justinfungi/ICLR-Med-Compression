"""
ACDC Dataset Loader
用于加载和处理ACDC心脏MRI数据集的PyTorch数据加载器

支持功能:
- 3D/4D数据加载  
- 关键时相数据提取
- 多种数据格式支持
- 数据预处理和增强
- 疾病分类标签处理

主要类:
- ACDCDataset: 核心数据集类
- ACDCDataModule: 便捷的数据模块包装器
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import SimpleITK as sitk
import pandas as pd
from scipy import ndimage
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="SimpleITK")

class ACDCDataset(Dataset):
    """
    ACDC心脏MRI数据集加载器
    
    支持多种加载模式:
    - 3D单帧数据 (ED/ES关键时相)
    - 4D时序数据 (完整心跳周期)
    - 分割标注数据
    - 患者元信息
    """
    
    # 疾病分类映射
    DISEASE_MAPPING = {
        'NOR': 0,   # 正常
        'DCM': 1,   # 扩张性心肌病
        'HCM': 2,   # 肥厚性心肌病  
        'ARV': 3,   # 异常右心室
        'MINF': 4   # 心梗后改变
    }
    
    # 分割标签映射
    SEGMENTATION_LABELS = {
        'background': 0,      # 背景
        'rv_cavity': 1,       # 右心室腔
        'lv_myocardium': 2,   # 左心室心肌
        'lv_cavity': 3        # 左心室腔
    }
    
    def __init__(
        self, 
        data_root: Union[str, Path],
        split: str = 'training',
        mode: str = '3d_keyframes',
        load_segmentation: bool = True,
        transform: Optional[callable] = None,
        target_spacing: Optional[Tuple[float, ...]] = None,
        normalize: bool = True,
        cache_data: bool = False
    ):
        """
        初始化ACDC数据集
        
        Args:
            data_root: 数据集根目录路径
            split: 数据集分割 ('training', 'testing')
            mode: 加载模式 ('3d_keyframes', '4d_sequence', 'ed_only', 'es_only')
            load_segmentation: 是否加载分割标注
            transform: 数据变换函数
            target_spacing: 目标体素间距 (x, y, z)
            normalize: 是否进行强度归一化
            cache_data: 是否缓存数据到内存
        """
        self.data_root = Path(data_root)
        self.split = split
        self.mode = mode
        self.load_segmentation = load_segmentation
        self.transform = transform
        self.target_spacing = target_spacing
        self.normalize = normalize
        self.cache_data = cache_data
        
        # 数据缓存
        self._cache = {} if cache_data else None
        
        # 扫描数据文件
        self.patient_list = self._scan_patients()
        self.patient_info = self._load_patient_info()
        
        print(f"✅ 加载ACDC {split}数据集: {len(self.patient_list)}例患者")
        print(f"📊 加载模式: {mode}")
        print(f"🎯 分割标注: {'开启' if load_segmentation else '关闭'}")
    
    def _scan_patients(self) -> List[str]:
        """扫描患者文件夹"""
        split_dir = self.data_root / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"数据集目录不存在: {split_dir}")
        
        # 获取所有患者文件夹
        patient_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('patient')]
        patient_list = sorted([d.name for d in patient_dirs])
        
        return patient_list
    
    def _load_patient_info(self) -> Dict[str, Dict]:
        """加载所有患者的基本信息"""
        patient_info = {}
        
        for patient_id in self.patient_list:
            info_file = self.data_root / self.split / patient_id / 'Info.cfg'
            if info_file.exists():
                info = self._parse_info_file(info_file)
                patient_info[patient_id] = info
            else:
                warnings.warn(f"Info.cfg文件不存在: {patient_id}")
                patient_info[patient_id] = {}
        
        return patient_info
    
    def _parse_info_file(self, info_file: Path) -> Dict:
        """解析患者信息文件"""
        info = {}
        with open(info_file, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 类型转换
                    if key in ['ED', 'ES', 'NbFrame']:
                        info[key] = int(value)
                    elif key in ['Height', 'Weight']:
                        info[key] = float(value)
                    else:
                        info[key] = value
        
        return info
    
    def _load_nifti(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
        """加载NIfTI文件"""
        if not filepath.exists():
            raise FileNotFoundError(f"文件不存在: {filepath}")
        
        # 使用SimpleITK加载
        image = sitk.ReadImage(str(filepath))
        
        # 提取图像数组
        image_array = sitk.GetArrayFromImage(image)
        
        # 提取元信息
        metadata = {
            'origin': image.GetOrigin(),
            'spacing': image.GetSpacing(),
            'direction': image.GetDirection(),
            'size': image.GetSize()
        }
        
        return image_array, metadata
    
    def _resample_image(self, image: np.ndarray, original_spacing: Tuple, target_spacing: Tuple) -> np.ndarray:
        """重采样图像到目标分辨率"""
        if len(original_spacing) != len(target_spacing):
            raise ValueError(f"spacing维度不匹配: {len(original_spacing)} vs {len(target_spacing)}")
        
        # 计算缩放因子
        zoom_factors = [orig / target for orig, target in zip(original_spacing, target_spacing)]
        
        # 执行重采样
        if image.dtype == np.int16 or image.dtype == np.int32:
            # 对于标签数据使用最近邻插值
            resampled = ndimage.zoom(image, zoom_factors, order=0)
        else:
            # 对于图像数据使用线性插值
            resampled = ndimage.zoom(image, zoom_factors, order=1)
        
        return resampled
    
    def _normalize_intensity(self, image: np.ndarray, method: str = 'z_score') -> np.ndarray:
        """强度归一化"""
        if method == 'z_score':
            # Z-score归一化
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                normalized = (image - mean) / std
            else:
                normalized = image - mean
        elif method == 'min_max':
            # Min-Max归一化到[0,1]
            min_val = np.min(image)
            max_val = np.max(image)
            if max_val > min_val:
                normalized = (image - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(image)
        else:
            normalized = image
        
        return normalized.astype(np.float32)
    
    def _get_patient_files(self, patient_id: str) -> Dict[str, Path]:
        """获取患者的所有文件路径"""
        patient_dir = self.data_root / self.split / patient_id
        files = {}
        
        # 4D序列文件
        files['4d'] = patient_dir / f"{patient_id}_4d.nii.gz"
        
        # 关键时相文件
        info = self.patient_info.get(patient_id, {})
        ed_frame = info.get('ED', 1)
        es_frame = info.get('ES', 14)
        
        files['ed_image'] = patient_dir / f"{patient_id}_frame{ed_frame:02d}.nii.gz"
        files['es_image'] = patient_dir / f"{patient_id}_frame{es_frame:02d}.nii.gz"
        
        if self.load_segmentation:
            files['ed_seg'] = patient_dir / f"{patient_id}_frame{ed_frame:02d}_gt.nii.gz"
            files['es_seg'] = patient_dir / f"{patient_id}_frame{es_frame:02d}_gt.nii.gz"
        
        return files
    
    def _load_patient_data(self, patient_id: str) -> Dict[str, Any]:
        """加载单个患者的数据"""
        # 检查缓存
        if self._cache is not None and patient_id in self._cache:
            return self._cache[patient_id]
        
        files = self._get_patient_files(patient_id)
        data = {'patient_id': patient_id}
        
        # 加载不同模式的数据
        if self.mode == '4d_sequence':
            # 加载4D时序数据
            if files['4d'].exists():
                image_4d, metadata = self._load_nifti(files['4d'])
                data['image'] = image_4d
                data['metadata'] = metadata
            else:
                raise FileNotFoundError(f"4D文件不存在: {files['4d']}")
        
        elif self.mode == '3d_keyframes':
            # 加载ED和ES关键帧
            images, segmentations = [], []
            
            for phase, img_file in [('ed', files['ed_image']), ('es', files['es_image'])]:
                if img_file.exists():
                    image, metadata = self._load_nifti(img_file)
                    images.append(image)
                    
                    # 加载分割标注
                    if self.load_segmentation:
                        seg_file = files[f'{phase}_seg']
                        if seg_file.exists():
                            seg, _ = self._load_nifti(seg_file)
                            segmentations.append(seg)
                        else:
                            warnings.warn(f"分割文件不存在: {seg_file}")
                            segmentations.append(np.zeros_like(image))
            
            data['images'] = np.stack(images) if images else None
            data['segmentations'] = np.stack(segmentations) if segmentations else None
            data['metadata'] = metadata if 'metadata' in locals() else {}
        
        elif self.mode in ['ed_only', 'es_only']:
            # 只加载ED或ES
            phase = self.mode.split('_')[0]
            img_file = files[f'{phase}_image']
            
            if img_file.exists():
                image, metadata = self._load_nifti(img_file)
                data['image'] = image
                data['metadata'] = metadata
                
                if self.load_segmentation:
                    seg_file = files[f'{phase}_seg']
                    if seg_file.exists():
                        seg, _ = self._load_nifti(seg_file)
                        data['segmentation'] = seg
        
        # 数据预处理
        data = self._preprocess_data(data)
        
        # 添加患者信息
        data['patient_info'] = self.patient_info.get(patient_id, {})
        
        # 添加疾病标签
        disease = data['patient_info'].get('Group', 'NOR')
        data['disease_label'] = self.DISEASE_MAPPING.get(disease, 0)
        
        # 缓存数据
        if self._cache is not None:
            self._cache[patient_id] = data
        
        return data
    
    def _preprocess_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """数据预处理"""
        # 重采样
        if self.target_spacing is not None and 'metadata' in data:
            original_spacing = data['metadata']['spacing']
            
            for key in ['image', 'images', 'segmentation', 'segmentations']:
                if key in data and data[key] is not None:
                    if key in ['image', 'segmentation']:
                        # 单个图像
                        data[key] = self._resample_image(
                            data[key], original_spacing, self.target_spacing
                        )
                    else:
                        # 图像序列
                        resampled = []
                        for img in data[key]:
                            resampled.append(self._resample_image(
                                img, original_spacing, self.target_spacing
                            ))
                        data[key] = np.stack(resampled)
        
        # 强度归一化
        if self.normalize:
            for key in ['image', 'images']:
                if key in data and data[key] is not None:
                    if key == 'image':
                        data[key] = self._normalize_intensity(data[key])
                    else:
                        normalized = []
                        for img in data[key]:
                            normalized.append(self._normalize_intensity(img))
                        data[key] = np.stack(normalized)
        
        return data
    
    def __len__(self) -> int:
        return len(self.patient_list)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """获取单个样本"""
        patient_id = self.patient_list[idx]
        data = self._load_patient_data(patient_id)
        
        # 应用数据变换
        if self.transform is not None:
            data = self.transform(data)
        
        # 转换为PyTorch张量
        for key in ['image', 'images', 'segmentation', 'segmentations']:
            if key in data and data[key] is not None:
                data[key] = torch.from_numpy(data[key]).float()
                if key in ['segmentation', 'segmentations']:
                    data[key] = data[key].long()
        
        data['disease_label'] = torch.tensor(data['disease_label'], dtype=torch.long)
        
        return data
    
    def get_patient_info(self, patient_id: str) -> Dict:
        """获取患者信息"""
        return self.patient_info.get(patient_id, {})
    
    def get_disease_distribution(self) -> Dict[str, int]:
        """获取疾病分布统计"""
        distribution = {}
        for patient_id in self.patient_list:
            info = self.patient_info.get(patient_id, {})
            disease = info.get('Group', 'NOR')
            distribution[disease] = distribution.get(disease, 0) + 1
        return distribution


class ACDCDataModule:
    """
    ACDC数据模块，提供便捷的数据加载接口
    """
    
    def __init__(
        self,
        data_root: Union[str, Path],
        batch_size: int = 4,
        num_workers: int = 4,
        mode: str = '3d_keyframes',
        target_spacing: Optional[Tuple[float, ...]] = None,
        normalize: bool = True,
        cache_data: bool = False
    ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.target_spacing = target_spacing
        self.normalize = normalize
        self.cache_data = cache_data
        
        self.train_dataset = None
        self.test_dataset = None
    
    def setup(self):
        """设置数据集"""
        # 训练集
        self.train_dataset = ACDCDataset(
            data_root=self.data_root,
            split='training',
            mode=self.mode,
            target_spacing=self.target_spacing,
            normalize=self.normalize,
            cache_data=self.cache_data
        )
        
        # 测试集
        self.test_dataset = ACDCDataset(
            data_root=self.data_root,
            split='testing',
            mode=self.mode,
            load_segmentation=False,  # 测试集通常没有标注
            target_spacing=self.target_spacing,
            normalize=self.normalize,
            cache_data=self.cache_data
        )
    
    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """测试数据加载器"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_statistics(self) -> Dict:
        """获取数据集统计信息"""
        if self.train_dataset is None:
            self.setup()
        
        stats = {
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'disease_distribution': self.train_dataset.get_disease_distribution(),
            'mode': self.mode,
            'target_spacing': self.target_spacing
        }
        
        return stats


def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    """
    自定义批处理函数，处理不同大小的图像
    """
    # 提取所有键
    keys = batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ['image', 'images', 'segmentation', 'segmentations']:
            # 图像数据需要特殊处理
            items = [item[key] for item in batch if item[key] is not None]
            if items:
                collated[key] = torch.stack(items)
            else:
                collated[key] = None
        elif key == 'disease_label':
            collated[key] = torch.tensor([item[key] for item in batch])
        else:
            collated[key] = [item[key] for item in batch]
    
    return collated


if __name__ == "__main__":
    # 使用示例
    data_root = "/Users/fenghaojie/Documents/ICLR/MedCompression/acdc_dataset"
    
    # 创建数据模块
    data_module = ACDCDataModule(
        data_root=data_root,
        batch_size=2,
        mode='3d_keyframes',
        target_spacing=(1.5, 1.5, 10.0),
        normalize=True
    )
    
    # 设置数据集
    data_module.setup()
    
    # 获取统计信息
    stats = data_module.get_statistics()
    print("📊 数据集统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 获取数据加载器
    train_loader = data_module.train_dataloader()
    
    # 测试数据加载
    print("\n🔍 测试数据加载:")
    for i, batch in enumerate(train_loader):
        print(f"Batch {i}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value)}")
        
        if i >= 2:  # 只测试前3个batch
            break
