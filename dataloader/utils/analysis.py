"""
ACDC数据集分析工具
提供数据集统计分析和报告生成
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


def analyze_dataset_statistics(dataset) -> Dict[str, Any]:
    """
    分析数据集统计信息
    
    Args:
        dataset: ACDCDataset实例
        
    Returns:
        统计信息字典
    """
    stats = {
        'total_patients': len(dataset),
        'disease_distribution': dataset.get_disease_distribution(),
        'patient_demographics': {},
        'image_statistics': {}
    }
    
    # 分析患者人口统计学信息
    ages, heights, weights = [], [], []
    
    for patient_id in dataset.patient_list:
        info = dataset.get_patient_info(patient_id)
        if 'Height' in info:
            heights.append(info['Height'])
        if 'Weight' in info:
            weights.append(info['Weight'])
    
    if heights:
        stats['patient_demographics']['height'] = {
            'mean': np.mean(heights),
            'std': np.std(heights),
            'min': np.min(heights),
            'max': np.max(heights)
        }
    
    if weights:
        stats['patient_demographics']['weight'] = {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights)
        }
    
    # 分析图像统计信息
    image_shapes = []
    intensities = []
    
    for i in range(min(10, len(dataset))):  # 采样分析
        try:
            data = dataset[i]
            if 'image' in data:
                image_shapes.append(data['image'].shape)
                intensities.extend(data['image'].flatten().tolist())
            elif 'images' in data:
                for img in data['images']:
                    image_shapes.append(img.shape)
                    intensities.extend(img.flatten().tolist())
        except:
            continue
    
    if image_shapes:
        stats['image_statistics']['shapes'] = {
            'unique_shapes': list(set(image_shapes)),
            'most_common_shape': max(set(image_shapes), key=image_shapes.count)
        }
    
    if intensities:
        stats['image_statistics']['intensity'] = {
            'mean': np.mean(intensities),
            'std': np.std(intensities),
            'min': np.min(intensities),
            'max': np.max(intensities)
        }
    
    return stats


def create_dataset_report(dataset, output_dir: Path):
    """
    创建数据集报告
    
    Args:
        dataset: ACDCDataset实例
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 生成ACDC数据集报告...")
    
    # 1. 分析统计信息
    stats = analyze_dataset_statistics(dataset)
    
    # 2. 保存统计信息为JSON
    with open(output_dir / 'dataset_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    # 3. 创建markdown报告
    report_md = f"""# ACDC数据集分析报告

## 📊 基本信息
- **总患者数**: {stats['total_patients']}
- **数据集分割**: {dataset.split}
- **加载模式**: {dataset.mode}

## 🏥 疾病分布
"""
    
    for disease, count in stats['disease_distribution'].items():
        percentage = count / stats['total_patients'] * 100
        report_md += f"- **{disease}**: {count}例 ({percentage:.1f}%)\n"
    
    report_md += f"""
## 👥 患者人口统计学信息
"""
    
    if 'height' in stats['patient_demographics']:
        height_data = stats['patient_demographics']['height']
        report_md += f"""
### 身高
- 平均值: {height_data['mean']:.1f} cm
- 标准差: {height_data['std']:.1f} cm
- 范围: {height_data['min']:.1f} - {height_data['max']:.1f} cm
"""
    
    if 'weight' in stats['patient_demographics']:
        weight_data = stats['patient_demographics']['weight']
        report_md += f"""
### 体重
- 平均值: {weight_data['mean']:.1f} kg
- 标准差: {weight_data['std']:.1f} kg
- 范围: {weight_data['min']:.1f} - {weight_data['max']:.1f} kg
"""
    
    if 'image_statistics' in stats:
        img_stats = stats['image_statistics']
        report_md += f"""
## 🖼️ 图像统计信息
"""
        if 'shapes' in img_stats:
            report_md += f"- **常见图像尺寸**: {img_stats['shapes']['most_common_shape']}\n"
        
        if 'intensity' in img_stats:
            intensity_data = img_stats['intensity']
            report_md += f"""- **强度统计**:
  - 平均值: {intensity_data['mean']:.3f}
  - 标准差: {intensity_data['std']:.3f}
  - 范围: {intensity_data['min']:.3f} - {intensity_data['max']:.3f}
"""
    
    # 保存markdown报告
    with open(output_dir / 'dataset_report.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    print(f"✅ 报告已生成: {output_dir}")
    print(f"📄 查看报告: {output_dir / 'dataset_report.md'}")


def get_dataset_summary(dataset) -> str:
    """
    获取数据集简要摘要
    
    Args:
        dataset: ACDCDataset实例
        
    Returns:
        摘要字符串
    """
    stats = analyze_dataset_statistics(dataset)
    
    summary = f"""
📊 ACDC数据集摘要
==================
总患者数: {stats['total_patients']}
数据分割: {dataset.split}
加载模式: {dataset.mode}

疾病分布:
"""
    
    for disease, count in stats['disease_distribution'].items():
        percentage = count / stats['total_patients'] * 100
        summary += f"  {disease}: {count}例 ({percentage:.1f}%)\n"
    
    return summary
