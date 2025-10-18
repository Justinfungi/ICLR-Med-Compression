"""
ACDC数据集专用变换函数
提供各种数据增强和预处理变换
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, List
from scipy import ndimage
import random


class ACDCTransform:
    """ACDC数据变换基类"""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class RandomRotation3D(ACDCTransform):
    """3D随机旋转"""
    
    def __init__(self, angle_range: float = 15.0, probability: float = 0.5):
        self.angle_range = angle_range
        self.probability = probability
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.probability:
            return data
        
        # 生成随机旋转角度
        angles = [random.uniform(-self.angle_range, self.angle_range) for _ in range(3)]
        
        for key in ['image', 'images', 'segmentation', 'segmentations']:
            if key in data and data[key] is not None:
                if key in ['image', 'segmentation']:
                    data[key] = self._rotate_3d(data[key], angles, key.endswith('segmentation'))
                else:
                    rotated = []
                    for img in data[key]:
                        rotated.append(self._rotate_3d(img, angles, key.endswith('segmentations')))
                    data[key] = np.stack(rotated)
        
        return data
    
    def _rotate_3d(self, image: np.ndarray, angles: List[float], is_label: bool = False) -> np.ndarray:
        """执行3D旋转"""
        order = 0 if is_label else 1  # 标签用最近邻，图像用线性插值
        
        # 依次进行三个轴的旋转
        for axis, angle in enumerate(angles):
            if abs(angle) > 0.1:  # 避免微小旋转
                image = ndimage.rotate(image, angle, axes=(1, 2), reshape=False, order=order)
        
        return image


class RandomFlip3D(ACDCTransform):
    """3D随机翻转"""
    
    def __init__(self, probability: float = 0.5, axes: List[int] = [1, 2]):
        self.probability = probability
        self.axes = axes
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.probability:
            return data
        
        # 随机选择翻转轴
        flip_axis = random.choice(self.axes)
        
        for key in ['image', 'images', 'segmentation', 'segmentations']:
            if key in data and data[key] is not None:
                if key in ['image', 'segmentation']:
                    data[key] = np.flip(data[key], axis=flip_axis).copy()
                else:
                    flipped = []
                    for img in data[key]:
                        flipped.append(np.flip(img, axis=flip_axis).copy())
                    data[key] = np.stack(flipped)
        
        return data


class RandomNoise(ACDCTransform):
    """添加随机噪声"""
    
    def __init__(self, noise_std: float = 0.1, probability: float = 0.3):
        self.noise_std = noise_std
        self.probability = probability
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.probability:
            return data
        
        for key in ['image', 'images']:
            if key in data and data[key] is not None:
                noise = np.random.normal(0, self.noise_std, data[key].shape)
                data[key] = data[key] + noise.astype(data[key].dtype)
        
        return data


class RandomIntensityShift(ACDCTransform):
    """随机强度偏移"""
    
    def __init__(self, shift_range: float = 0.2, probability: float = 0.4):
        self.shift_range = shift_range
        self.probability = probability
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.probability:
            return data
        
        for key in ['image', 'images']:
            if key in data and data[key] is not None:
                shift = random.uniform(-self.shift_range, self.shift_range)
                data[key] = data[key] + shift
        
        return data


class RandomScale(ACDCTransform):
    """随机缩放"""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1), probability: float = 0.3):
        self.scale_range = scale_range
        self.probability = probability
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.probability:
            return data
        
        scale = random.uniform(*self.scale_range)
        zoom_factors = [1.0, scale, scale]  # 只缩放x,y维度，保持z维度
        
        for key in ['image', 'images', 'segmentation', 'segmentations']:
            if key in data and data[key] is not None:
                is_label = key.endswith('segmentation') or key.endswith('segmentations')
                order = 0 if is_label else 1
                
                if key in ['image', 'segmentation']:
                    data[key] = ndimage.zoom(data[key], zoom_factors, order=order)
                else:
                    scaled = []
                    for img in data[key]:
                        scaled.append(ndimage.zoom(img, zoom_factors, order=order))
                    data[key] = np.stack(scaled)
        
        return data


class CenterCrop3D(ACDCTransform):
    """3D中心裁剪"""
    
    def __init__(self, output_size: Tuple[int, int, int]):
        self.output_size = output_size
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in ['image', 'images', 'segmentation', 'segmentations']:
            if key in data and data[key] is not None:
                if key in ['image', 'segmentation']:
                    data[key] = self._center_crop_3d(data[key])
                else:
                    cropped = []
                    for img in data[key]:
                        cropped.append(self._center_crop_3d(img))
                    data[key] = np.stack(cropped)
        
        return data
    
    def _center_crop_3d(self, image: np.ndarray) -> np.ndarray:
        """执行3D中心裁剪"""
        d, h, w = image.shape
        td, th, tw = self.output_size
        
        # 计算裁剪起始位置
        d_start = max(0, (d - td) // 2)
        h_start = max(0, (h - th) // 2)
        w_start = max(0, (w - tw) // 2)
        
        # 执行裁剪
        cropped = image[
            d_start:d_start + td,
            h_start:h_start + th,
            w_start:w_start + tw
        ]
        
        # 如果原图像小于目标大小，进行零填充
        if cropped.shape != self.output_size:
            padded = np.zeros(self.output_size, dtype=image.dtype)
            
            # 计算填充位置
            pad_d = (self.output_size[0] - cropped.shape[0]) // 2
            pad_h = (self.output_size[1] - cropped.shape[1]) // 2
            pad_w = (self.output_size[2] - cropped.shape[2]) // 2
            
            padded[
                pad_d:pad_d + cropped.shape[0],
                pad_h:pad_h + cropped.shape[1],
                pad_w:pad_w + cropped.shape[2]
            ] = cropped
            
            return padded
        
        return cropped


class Pad3D(ACDCTransform):
    """3D零填充"""
    
    def __init__(self, output_size: Tuple[int, int, int], mode: str = 'constant'):
        self.output_size = output_size
        self.mode = mode
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in ['image', 'images', 'segmentation', 'segmentations']:
            if key in data and data[key] is not None:
                if key in ['image', 'segmentation']:
                    data[key] = self._pad_3d(data[key])
                else:
                    padded = []
                    for img in data[key]:
                        padded.append(self._pad_3d(img))
                    data[key] = np.stack(padded)
        
        return data
    
    def _pad_3d(self, image: np.ndarray) -> np.ndarray:
        """执行3D填充"""
        d, h, w = image.shape
        td, th, tw = self.output_size
        
        if d >= td and h >= th and w >= tw:
            return image  # 不需要填充
        
        # 计算填充量
        pad_d = max(0, td - d)
        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        
        # 计算填充配置
        pad_width = [
            (pad_d // 2, pad_d - pad_d // 2),
            (pad_h // 2, pad_h - pad_h // 2),
            (pad_w // 2, pad_w - pad_w // 2)
        ]
        
        return np.pad(image, pad_width, mode=self.mode)


class ToTensor(ACDCTransform):
    """转换为PyTorch张量"""
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for key in ['image', 'images', 'segmentation', 'segmentations']:
            if key in data and data[key] is not None:
                tensor = torch.from_numpy(data[key]).float()
                if key.endswith('segmentation') or key.endswith('segmentations'):
                    tensor = tensor.long()
                data[key] = tensor
        
        if 'disease_label' in data:
            data['disease_label'] = torch.tensor(data['disease_label'], dtype=torch.long)
        
        return data


class Compose(ACDCTransform):
    """组合多个变换"""
    
    def __init__(self, transforms: List[ACDCTransform]):
        self.transforms = transforms
    
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data


# 预定义的变换组合
def get_train_transforms(
    output_size: Optional[Tuple[int, int, int]] = None,
    augmentation: bool = True
) -> Compose:
    """获取训练时的数据变换"""
    transforms = []
    
    if augmentation:
        transforms.extend([
            RandomRotation3D(angle_range=10.0, probability=0.3),
            RandomFlip3D(probability=0.5),
            RandomNoise(noise_std=0.05, probability=0.2),
            RandomIntensityShift(shift_range=0.1, probability=0.3),
            RandomScale(scale_range=(0.95, 1.05), probability=0.2)
        ])
    
    if output_size is not None:
        transforms.append(CenterCrop3D(output_size))
    
    return Compose(transforms)


def get_val_transforms(
    output_size: Optional[Tuple[int, int, int]] = None
) -> Compose:
    """获取验证时的数据变换"""
    transforms = []
    
    if output_size is not None:
        transforms.append(CenterCrop3D(output_size))
    
    return Compose(transforms)


def get_test_transforms(
    output_size: Optional[Tuple[int, int, int]] = None
) -> Compose:
    """获取测试时的数据变换"""
    return get_val_transforms(output_size)
