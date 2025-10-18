"""
ACDC数据集评估指标
提供心脏功能指标计算和分割评估
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def calculate_cardiac_metrics(ed_seg: np.ndarray, es_seg: np.ndarray, 
                             spacing: Tuple[float, float, float]) -> Dict[str, float]:
    """
    计算心脏功能指标
    
    Args:
        ed_seg: ED时相分割图
        es_seg: ES时相分割图
        spacing: 体素间距 (x, y, z)
        
    Returns:
        心脏功能指标字典
    """
    # 体素体积
    voxel_volume = np.prod(spacing)  # mm³
    
    # 计算各结构体积
    def calculate_volume(seg_map: np.ndarray, label: int) -> float:
        """计算指定标签的体积"""
        voxel_count = np.sum(seg_map == label)
        return voxel_count * voxel_volume / 1000  # 转换为ml
    
    # ED时相体积
    ed_lv_cavity = calculate_volume(ed_seg, 3)  # 左心室腔
    ed_rv_cavity = calculate_volume(ed_seg, 1)  # 右心室腔
    ed_lv_myocardium = calculate_volume(ed_seg, 2)  # 左心室心肌
    
    # ES时相体积
    es_lv_cavity = calculate_volume(es_seg, 3)
    es_rv_cavity = calculate_volume(es_seg, 1)
    es_lv_myocardium = calculate_volume(es_seg, 2)
    
    # 计算功能指标
    metrics = {
        # 左心室指标
        'lv_edv': ed_lv_cavity,  # 左心室舒张末期容积
        'lv_esv': es_lv_cavity,  # 左心室收缩末期容积
        'lv_sv': ed_lv_cavity - es_lv_cavity,  # 左心室每搏量
        'lv_ef': (ed_lv_cavity - es_lv_cavity) / ed_lv_cavity * 100 if ed_lv_cavity > 0 else 0,  # 左心室射血分数
        
        # 右心室指标
        'rv_edv': ed_rv_cavity,
        'rv_esv': es_rv_cavity,
        'rv_sv': ed_rv_cavity - es_rv_cavity,
        'rv_ef': (ed_rv_cavity - es_rv_cavity) / ed_rv_cavity * 100 if ed_rv_cavity > 0 else 0,
        
        # 心肌质量
        'lv_myocardium_mass': (ed_lv_myocardium + es_lv_myocardium) / 2 * 1.05  # 假设心肌密度1.05g/ml
    }
    
    return metrics


def evaluate_segmentation(pred_seg: np.ndarray, true_seg: np.ndarray) -> Dict[str, float]:
    """
    评估分割结果
    
    Args:
        pred_seg: 预测分割图
        true_seg: 真实分割图
        
    Returns:
        评估指标字典
    """
    metrics = {}
    
    # 计算每个类别的Dice系数
    for label in [1, 2, 3]:  # RV, LV心肌, LV腔
        pred_mask = (pred_seg == label).astype(bool)
        true_mask = (true_seg == label).astype(bool)
        
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = pred_mask.sum() + true_mask.sum()
        
        if union > 0:
            dice = 2 * intersection / union
        else:
            dice = 1.0 if intersection == 0 else 0.0
        
        label_names = {1: 'rv', 2: 'lv_myo', 3: 'lv'}
        metrics[f'dice_{label_names[label]}'] = dice
    
    # 计算平均Dice
    dice_scores = [metrics[k] for k in metrics.keys() if k.startswith('dice_')]
    metrics['dice_mean'] = np.mean(dice_scores)
    
    return metrics


def calculate_hausdorff_distance(pred_seg: np.ndarray, true_seg: np.ndarray, 
                                spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> Dict[str, float]:
    """
    计算Hausdorff距离
    
    Args:
        pred_seg: 预测分割图
        true_seg: 真实分割图
        spacing: 体素间距
        
    Returns:
        Hausdorff距离字典
    """
    try:
        from scipy.spatial.distance import directed_hausdorff
    except ImportError:
        print("警告: 需要安装scipy来计算Hausdorff距离")
        return {}
    
    metrics = {}
    
    for label in [1, 2, 3]:  # RV, LV心肌, LV腔
        pred_mask = (pred_seg == label)
        true_mask = (true_seg == label)
        
        if not (pred_mask.any() and true_mask.any()):
            continue
        
        # 获取边界点
        pred_points = np.array(np.where(pred_mask)).T * np.array(spacing)
        true_points = np.array(np.where(true_mask)).T * np.array(spacing)
        
        # 计算双向Hausdorff距离
        hd1 = directed_hausdorff(pred_points, true_points)[0]
        hd2 = directed_hausdorff(true_points, pred_points)[0]
        hd = max(hd1, hd2)
        
        label_names = {1: 'rv', 2: 'lv_myo', 3: 'lv'}
        metrics[f'hd_{label_names[label]}'] = hd
    
    return metrics


def calculate_volume_similarity(pred_seg: np.ndarray, true_seg: np.ndarray,
                               spacing: Tuple[float, float, float]) -> Dict[str, float]:
    """
    计算体积相似性指标
    
    Args:
        pred_seg: 预测分割图
        true_seg: 真实分割图
        spacing: 体素间距
        
    Returns:
        体积相似性指标字典
    """
    voxel_volume = np.prod(spacing) / 1000  # 转换为ml
    metrics = {}
    
    for label in [1, 2, 3]:  # RV, LV心肌, LV腔
        pred_volume = np.sum(pred_seg == label) * voxel_volume
        true_volume = np.sum(true_seg == label) * voxel_volume
        
        if true_volume > 0:
            volume_error = abs(pred_volume - true_volume) / true_volume * 100
        else:
            volume_error = 0.0 if pred_volume == 0 else 100.0
        
        label_names = {1: 'rv', 2: 'lv_myo', 3: 'lv'}
        metrics[f'volume_error_{label_names[label]}'] = volume_error
        metrics[f'pred_volume_{label_names[label]}'] = pred_volume
        metrics[f'true_volume_{label_names[label]}'] = true_volume
    
    return metrics


def evaluate_cardiac_function(pred_metrics: Dict[str, float], 
                             true_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    评估心脏功能指标的预测准确性
    
    Args:
        pred_metrics: 预测的心脏功能指标
        true_metrics: 真实的心脏功能指标
        
    Returns:
        功能指标评估结果
    """
    evaluation = {}
    
    key_metrics = ['lv_ef', 'rv_ef', 'lv_edv', 'lv_esv', 'lv_sv']
    
    for metric in key_metrics:
        if metric in pred_metrics and metric in true_metrics:
            pred_val = pred_metrics[metric]
            true_val = true_metrics[metric]
            
            # 绝对误差
            abs_error = abs(pred_val - true_val)
            
            # 相对误差
            if true_val != 0:
                rel_error = abs_error / abs(true_val) * 100
            else:
                rel_error = 0.0 if pred_val == 0 else 100.0
            
            evaluation[f'{metric}_abs_error'] = abs_error
            evaluation[f'{metric}_rel_error'] = rel_error
    
    return evaluation


def assess_cardiac_health(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    评估心脏健康状态
    
    Args:
        metrics: 心脏功能指标
        
    Returns:
        健康状态评估
    """
    assessment = {}
    
    # 左心室射血分数评估
    lv_ef = metrics.get('lv_ef', 0)
    if lv_ef >= 55:
        assessment['lv_function'] = '正常'
    elif lv_ef >= 45:
        assessment['lv_function'] = '轻度减低'
    elif lv_ef >= 35:
        assessment['lv_function'] = '中度减低'
    else:
        assessment['lv_function'] = '重度减低'
    
    # 右心室射血分数评估
    rv_ef = metrics.get('rv_ef', 0)
    if rv_ef >= 45:
        assessment['rv_function'] = '正常'
    elif rv_ef >= 35:
        assessment['rv_function'] = '轻度减低'
    else:
        assessment['rv_function'] = '减低'
    
    # 左心室容积评估
    lv_edv = metrics.get('lv_edv', 0)
    if lv_edv <= 140:  # ml, 男性正常上限
        assessment['lv_size'] = '正常'
    elif lv_edv <= 180:
        assessment['lv_size'] = '轻度扩大'
    elif lv_edv <= 220:
        assessment['lv_size'] = '中度扩大'
    else:
        assessment['lv_size'] = '重度扩大'
    
    return assessment
