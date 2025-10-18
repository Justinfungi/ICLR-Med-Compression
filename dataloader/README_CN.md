# 🫀 ACDC心脏MRI数据加载器

PyTorch实现的ACDC心脏MRI数据集加载器，支持多模态数据加载、预处理和分析。

## ✨ 核心特性

- **多模态加载**: 3D关键帧 / 4D时序 / 单帧模式
- **智能预处理**: 自动重采样、归一化、数据增强
- **完整工具链**: 分析统计、可视化、心脏功能指标计算
- **即用即走**: 提供完整示例代码，开箱即用

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 基本使用

```python
from acdc_dataset import ACDCDataset, ACDCDataModule

# 加载数据集
dataset = ACDCDataset(
    data_root="path/to/acdc_dataset",
    split='training',
    mode='3d_keyframes',        # ED和ES关键帧
    load_segmentation=True,     # 加载分割标注
    normalize=True              # 强度归一化
)

# 获取样本
sample = dataset[0]
print(f"图像: {sample['images'].shape}")      # [2, Z, H, W]
print(f"分割: {sample['segmentations'].shape}")
print(f"疾病: {sample['patient_info']['Group']}")
```

## 📊 数据模式

| 模式 | 说明 | 输出形状 | 应用场景 |
|------|------|----------|----------|
| `3d_keyframes` | ED/ES关键帧 | `[2, Z, H, W]` | 分割、功能评估 |
| `4d_sequence` | 完整心跳周期 | `[T, Z, H, W]` | 运动分析 |
| `ed_only` | 舒张末期 | `[Z, H, W]` | 分类任务 |
| `es_only` | 收缩末期 | `[Z, H, W]` | 分类任务 |

## 🎯 典型应用

### 1. 心脏分割

```python
from utils.transforms import get_train_transforms

dataset = ACDCDataset(
    data_root="path/to/acdc_dataset",
    mode='3d_keyframes',
    load_segmentation=True,
    transform=get_train_transforms(
        output_size=(10, 256, 256),
        augmentation=True
    )
)
```

### 2. 疾病分类

```python
dataset = ACDCDataset(
    mode='ed_only',
    load_segmentation=False
)

# 获取疾病标签 (NOR=0, DCM=1, HCM=2, ARV=3, MINF=4)
disease_label = dataset[0]['disease_label']
```

### 3. 心脏功能评估

```python
from utils.metrics import calculate_cardiac_metrics

sample = dataset[0]
metrics = calculate_cardiac_metrics(
    sample['segmentations'][0].numpy(),  # ED
    sample['segmentations'][1].numpy(),  # ES
    sample['metadata']['spacing']
)

print(f"LVEF: {metrics['lv_ef']:.1f}%")
print(f"RVEF: {metrics['rv_ef']:.1f}%")
```

## 🔧 高级功能

### 数据增强

```python
from utils.transforms import RandomRotation3D, RandomFlip3D, Compose

transforms = Compose([
    RandomRotation3D(angle_range=15.0, probability=0.5),
    RandomFlip3D(probability=0.5),
    CenterCrop3D(output_size=(10, 256, 256))
])
```

### 数据分析

```python
from utils.analysis import create_dataset_report

# 生成完整数据集分析报告
create_dataset_report(dataset, output_dir="analysis_results")
```

### 可视化

```python
from utils.visualization import visualize_cardiac_phases

# 可视化心脏时相
sample = dataset[0]
visualize_cardiac_phases(sample, slice_idx=5)
```

## 📁 项目结构

```
dataloader/
├── acdc_dataset.py       # 核心数据集类
├── utils/
│   ├── transforms.py     # 数据变换
│   ├── metrics.py        # 评估指标
│   ├── analysis.py       # 数据分析
│   └── visualization.py  # 可视化工具
├── tests/
│   └── example_usage.py  # 完整使用示例
├── requirements.txt      # 依赖包
└── README_CN.md         # 本文档
```

## 📖 数据集信息

### 疾病类型
- **NOR**: 正常心脏
- **DCM**: 扩张性心肌病 (LVEF<40%)
- **HCM**: 肥厚性心肌病
- **ARV**: 异常右心室
- **MINF**: 心肌梗死后改变

### 分割标签
- `0`: 背景
- `1`: 右心室腔
- `2`: 左心室心肌
- `3`: 左心室腔

### 数据规格
- **格式**: NIfTI (.nii.gz)
- **分辨率**: ~1.5×1.5×10.0 mm
- **切片数**: 通常10个短轴切片
- **时间帧**: 28-30帧/心动周期

## 💡 运行示例

```bash
# 运行完整功能测试
cd tests
python example_usage.py

# 选择功能菜单，或运行所有示例
```

## 📚 引用

如使用本代码，请引用原始ACDC数据集论文:

```bibtex
@article{bernard2018deep,
  title={Deep learning techniques for automatic MRI cardiac multi-structures
         segmentation and diagnosis: is the problem solved?},
  author={Bernard, Olivier and Lalande, Alain and Zotti, Clement and
          Cervenansky, Frederic and others},
  journal={IEEE transactions on medical imaging},
  volume={37},
  number={11},
  pages={2514--2525},
  year={2018}
}
```

## ⚠️ 注意事项

1. **数据路径**: 确保数据集目录包含 `training/` 和 `testing/` 文件夹
2. **内存管理**: 大数据集建议关闭缓存 (`cache_data=False`)
3. **多进程**: Windows用户建议设置 `num_workers=0`

## 📞 支持

- 问题反馈: 提交Issue
- 功能建议: 提交PR
- 详细文档: 参见 `README.md`

## 📄 许可

代码: MIT License
数据集: CC BY-NC-SA 4.0 (仅限非商业科研用途)
