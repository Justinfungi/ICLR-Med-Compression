"""
ACDC数据集可视化工具
提供数据可视化和绘图功能
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path


def plot_disease_distribution(disease_dist: Dict[str, int], save_path: Optional[Path] = None):
    """
    绘制疾病分布图
    
    Args:
        disease_dist: 疾病分布字典
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    diseases = list(disease_dist.keys())
    counts = list(disease_dist.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(diseases)))
    
    # 绘制柱状图
    bars = plt.bar(diseases, counts, color=colors)
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.title('ACDC数据集疾病分布', fontsize=14, fontweight='bold')
    plt.xlabel('疾病类型')
    plt.ylabel('患者数量')
    plt.grid(axis='y', alpha=0.3)
    
    # 添加疾病全名
    disease_names = {
        'NOR': '正常',
        'DCM': '扩张性心肌病',
        'HCM': '肥厚性心肌病',
        'ARV': '异常右心室',
        'MINF': '心梗后改变'
    }
    
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(len(diseases)))
    ax2.set_xticklabels([disease_names.get(d, d) for d in diseases], rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_patient_demographics(stats: Dict[str, Any], save_path: Optional[Path] = None):
    """
    绘制患者人口统计学信息
    
    Args:
        stats: 数据集统计信息
        save_path: 保存路径
    """
    demographics = stats.get('patient_demographics', {})
    
    if not demographics:
        print("❌ 没有人口统计学数据可绘制")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 身高分布
    if 'height' in demographics:
        height_data = demographics['height']
        ax = axes[0]
        ax.text(0.5, 0.8, f"身高统计", ha='center', va='center', transform=ax.transAxes, 
                fontsize=12, fontweight='bold')
        ax.text(0.5, 0.6, f"平均值: {height_data['mean']:.1f} cm", ha='center', va='center', 
                transform=ax.transAxes)
        ax.text(0.5, 0.5, f"标准差: {height_data['std']:.1f} cm", ha='center', va='center', 
                transform=ax.transAxes)
        ax.text(0.5, 0.4, f"范围: {height_data['min']:.1f} - {height_data['max']:.1f} cm", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    # 体重分布
    if 'weight' in demographics:
        weight_data = demographics['weight']
        ax = axes[1]
        ax.text(0.5, 0.8, f"体重统计", ha='center', va='center', transform=ax.transAxes, 
                fontsize=12, fontweight='bold')
        ax.text(0.5, 0.6, f"平均值: {weight_data['mean']:.1f} kg", ha='center', va='center', 
                transform=ax.transAxes)
        ax.text(0.5, 0.5, f"标准差: {weight_data['std']:.1f} kg", ha='center', va='center', 
                transform=ax.transAxes)
        ax.text(0.5, 0.4, f"范围: {weight_data['min']:.1f} - {weight_data['max']:.1f} kg", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    plt.suptitle('ACDC数据集患者人口统计学信息', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_cardiac_phases(data: Dict[str, Any], slice_idx: Optional[int] = None):
    """
    可视化心脏时相
    
    Args:
        data: 数据样本
        slice_idx: 切片索引，None表示自动选择中间切片
    """
    if 'images' not in data or data['images'] is None:
        print("❌ 没有图像数据可可视化")
        return
    
    images = data['images']
    segmentations = data.get('segmentations', None)
    
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    if segmentations is not None and isinstance(segmentations, torch.Tensor):
        segmentations = segmentations.numpy()
    
    # 选择中间切片
    if slice_idx is None:
        slice_idx = images.shape[1] // 2
    
    # 创建子图
    n_cols = 4 if segmentations is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4))
    
    if n_cols == 2:
        axes = [axes[0], None, axes[1], None]
    
    # 显示ED相
    ed_img = images[0, slice_idx]
    axes[0].imshow(ed_img, cmap='gray')
    axes[0].set_title('ED (舒张末期)')
    axes[0].axis('off')
    
    if segmentations is not None:
        ed_seg = segmentations[0, slice_idx]
        axes[1].imshow(ed_seg, cmap='viridis')
        axes[1].set_title('ED分割标注')
        axes[1].axis('off')
    
    # 显示ES相
    es_img = images[1, slice_idx]
    axes[2].imshow(es_img, cmap='gray')
    axes[2].set_title('ES (收缩末期)')
    axes[2].axis('off')
    
    if segmentations is not None:
        es_seg = segmentations[1, slice_idx]
        axes[3].imshow(es_seg, cmap='viridis')
        axes[3].set_title('ES分割标注')
        axes[3].axis('off')
    
    # 添加患者信息
    patient_info = data.get('patient_info', {})
    disease = patient_info.get('Group', 'Unknown')
    plt.suptitle(f'患者: {data.get("patient_id", "Unknown")} | 疾病: {disease} | 切片: {slice_idx}', 
                 fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_cardiac_metrics_comparison(metrics_list: List[Dict[str, float]], 
                                   labels: List[str], 
                                   save_path: Optional[Path] = None):
    """
    绘制心脏功能指标对比图
    
    Args:
        metrics_list: 指标字典列表
        labels: 标签列表
        save_path: 保存路径
    """
    key_metrics = ['lv_ef', 'rv_ef', 'lv_edv', 'lv_esv']
    metric_names = {
        'lv_ef': 'LVEF (%)',
        'rv_ef': 'RVEF (%)', 
        'lv_edv': 'LVEDV (ml)',
        'lv_esv': 'LVESV (ml)'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(key_metrics):
        values = []
        for metrics in metrics_list:
            values.append(metrics.get(metric, 0))
        
        axes[i].bar(labels, values, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
        axes[i].set_title(metric_names[metric])
        axes[i].set_ylabel('值')
        
        # 添加数值标签
        for j, v in enumerate(values):
            axes[i].text(j, v + max(values)*0.01, f'{v:.1f}', 
                        ha='center', va='bottom')
    
    plt.suptitle('心脏功能指标对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_segmentation_overlay(image: np.ndarray, segmentation: np.ndarray, 
                             slice_idx: Optional[int] = None,
                             alpha: float = 0.5,
                             save_path: Optional[Path] = None):
    """
    绘制分割结果叠加图
    
    Args:
        image: 原始图像
        segmentation: 分割图
        slice_idx: 切片索引
        alpha: 透明度
        save_path: 保存路径
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    img_slice = image[slice_idx]
    seg_slice = segmentation[slice_idx]
    
    # 创建颜色映射
    colors = {
        0: [0, 0, 0, 0],      # 背景 - 透明
        1: [1, 1, 0, alpha],   # RV - 黄色
        2: [1, 0, 0, alpha],   # LV心肌 - 红色  
        3: [0, 0, 1, alpha]    # LV腔 - 蓝色
    }
    
    # 创建彩色分割图
    h, w = seg_slice.shape
    colored_seg = np.zeros((h, w, 4))
    
    for label, color in colors.items():
        mask = seg_slice == label
        colored_seg[mask] = color
    
    plt.figure(figsize=(10, 5))
    
    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(img_slice, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    # 分割图
    plt.subplot(1, 3, 2)
    plt.imshow(seg_slice, cmap='viridis')
    plt.title('分割标注')
    plt.axis('off')
    
    # 叠加图
    plt.subplot(1, 3, 3)
    plt.imshow(img_slice, cmap='gray')
    plt.imshow(colored_seg, alpha=alpha)
    plt.title('叠加显示')
    plt.axis('off')
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='yellow', label='右心室腔'),
        Patch(facecolor='red', label='左心室心肌'),
        Patch(facecolor='blue', label='左心室腔')
    ]
    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def create_cardiac_animation_frames(image_4d: np.ndarray, slice_idx: Optional[int] = None) -> List[np.ndarray]:
    """
    创建心跳动画帧
    
    Args:
        image_4d: 4D图像 (T, Z, H, W)
        slice_idx: 切片索引
        
    Returns:
        动画帧列表
    """
    if slice_idx is None:
        slice_idx = image_4d.shape[1] // 2
    
    frames = []
    for t in range(image_4d.shape[0]):
        frame = image_4d[t, slice_idx]
        frames.append(frame)
    
    return frames


def plot_intensity_histogram(images: np.ndarray, title: str = "强度分布", 
                           save_path: Optional[Path] = None):
    """
    绘制图像强度直方图
    
    Args:
        images: 图像数组
        title: 图标题
        save_path: 保存路径
    """
    plt.figure(figsize=(10, 6))
    
    # 展平所有像素值
    pixel_values = images.flatten()
    
    # 绘制直方图
    plt.hist(pixel_values, bins=100, alpha=0.7, density=True)
    plt.title(title)
    plt.xlabel('像素强度')
    plt.ylabel('密度')
    plt.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_val = np.mean(pixel_values)
    std_val = np.std(pixel_values)
    plt.axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.3f}')
    plt.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7, label=f'+1σ: {mean_val + std_val:.3f}')
    plt.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7, label=f'-1σ: {mean_val - std_val:.3f}')
    
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
