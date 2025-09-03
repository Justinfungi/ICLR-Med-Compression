# ACDC心脏MRI数据集加载器

这是一个专门为ACDC (Automated Cardiac Diagnosis Challenge) 心脏MRI数据集设计的PyTorch数据加载器，提供了完整的数据处理、增强和分析功能。

## 📋 功能特点

### 🎯 核心功能
- ✅ **多模式数据加载**: 支持3D/4D、单帧/时序、ED/ES等多种加载模式
- ✅ **完整预处理**: 自动重采样、强度归一化、数据类型转换
- ✅ **丰富的数据增强**: 旋转、翻转、噪声、缩放等3D医学图像专用变换
- ✅ **智能缓存**: 可选的内存缓存，加速重复访问
- ✅ **批量处理**: 优化的DataLoader，支持多进程加载

### 📊 分析工具
- ✅ **数据集统计**: 自动分析患者分布、图像特征等
- ✅ **可视化工具**: 心脏时相可视化、疾病分布图表
- ✅ **功能指标计算**: 自动计算LVEF、RVEF等心脏功能参数
- ✅ **报告生成**: 自动生成完整的数据集分析报告

## 🚀 快速开始

### 安装依赖

```bash
pip install torch torchvision numpy matplotlib seaborn
pip install SimpleITK scipy scikit-learn pandas pathlib
```

### 基本使用

```python
from acdc_dataset import ACDCDataset, ACDCDataModule
from acdc_transforms import get_train_transforms

# 1. 创建数据集
dataset = ACDCDataset(
    data_root="path/to/acdc_dataset",
    split='training',
    mode='3d_keyframes',  # ED和ES关键帧
    load_segmentation=True,
    normalize=True
)

# 2. 获取样本
sample = dataset[0]
print(f"图像形状: {sample['images'].shape}")  # [2, Z, H, W] (ED, ES)
print(f"分割形状: {sample['segmentations'].shape}")
print(f"疾病类型: {sample['patient_info']['Group']}")

# 3. 使用数据模块
data_module = ACDCDataModule(
    data_root="path/to/acdc_dataset",
    batch_size=4,
    mode='3d_keyframes'
)
data_module.setup()

train_loader = data_module.train_dataloader()
```

## 📚 详细文档

### 数据加载模式

| 模式 | 说明 | 输出格式 |
|------|------|----------|
| `'3d_keyframes'` | ED和ES关键帧 | `images: [2, Z, H, W]` |
| `'4d_sequence'` | 完整心跳周期 | `image: [T, Z, H, W]` |
| `'ed_only'` | 只加载ED帧 | `image: [Z, H, W]` |
| `'es_only'` | 只加载ES帧 | `image: [Z, H, W]` |

### 数据变换

```python
from acdc_transforms import get_train_transforms, get_val_transforms

# 训练时数据增强
train_transforms = get_train_transforms(
    output_size=(8, 224, 224),  # 统一输出尺寸
    augmentation=True  # 开启数据增强
)

# 验证时变换
val_transforms = get_val_transforms(
    output_size=(8, 224, 224)
)

dataset = ACDCDataset(
    data_root="path/to/acdc_dataset",
    transform=train_transforms
)
```

### 自定义变换

```python
from acdc_transforms import RandomRotation3D, RandomFlip3D, Compose

# 创建自定义变换
custom_transforms = Compose([
    RandomRotation3D(angle_range=15.0, probability=0.5),
    RandomFlip3D(probability=0.5),
    CenterCrop3D(output_size=(10, 256, 256))
])
```

## 🔧 高级功能

### 心脏功能指标计算

```python
from acdc_utils import calculate_cardiac_metrics

# 获取包含分割的样本
sample = dataset[0]
ed_seg = sample['segmentations'][0].numpy()
es_seg = sample['segmentations'][1].numpy()
spacing = sample['metadata']['spacing']

# 计算心脏功能指标
metrics = calculate_cardiac_metrics(ed_seg, es_seg, spacing)
print(f"左心室射血分数: {metrics['lv_ef']:.1f}%")
print(f"右心室射血分数: {metrics['rv_ef']:.1f}%")
```

### 数据集分析和可视化

```python
from acdc_utils import create_dataset_report, visualize_cardiac_phases

# 生成完整数据集报告
create_dataset_report(dataset, output_dir="analysis_results")

# 可视化心脏时相
sample = dataset[0]
visualize_cardiac_phases(sample, slice_idx=5)
```

### 4D时序数据处理

```python
# 加载4D数据
dataset_4d = ACDCDataset(
    data_root="path/to/acdc_dataset",
    mode='4d_sequence',
    load_segmentation=False
)

sample = dataset_4d[0]
print(f"4D图像形状: {sample['image'].shape}")  # [T, Z, H, W]
```

## 📊 数据集信息

### 疾病分类
- **NOR**: 正常心脏
- **DCM**: 扩张性心肌病
- **HCM**: 肥厚性心肌病
- **ARV**: 异常右心室
- **MINF**: 心肌梗死后改变

### 分割标签
- **0**: 背景
- **1**: 右心室腔
- **2**: 左心室心肌
- **3**: 左心室腔

### 数据格式
- **图像格式**: NIfTI (.nii.gz)
- **数据类型**: int16
- **空间分辨率**: 1.5625×1.5625×10.0 mm
- **时间分辨率**: ~33ms/帧

## 🎯 使用场景

### 1. 图像分割任务

```python
# 配置分割任务数据集
dataset = ACDCDataset(
    data_root="path/to/acdc_dataset",
    mode='3d_keyframes',
    load_segmentation=True,
    target_spacing=(1.5, 1.5, 8.0),  # 重采样
    transform=get_train_transforms(
        output_size=(10, 256, 256),
        augmentation=True
    )
)
```

### 2. 疾病分类任务

```python
# 配置分类任务数据集
dataset = ACDCDataset(
    data_root="path/to/acdc_dataset",
    mode='ed_only',  # 只用ED帧
    load_segmentation=False,
    normalize=True
)

# 获取疾病标签
sample = dataset[0]
disease_label = sample['disease_label']  # 0-4对应不同疾病
```

### 3. 心脏功能评估

```python
# 加载关键帧用于功能评估
dataset = ACDCDataset(
    mode='3d_keyframes',
    load_segmentation=True
)

for sample in dataset:
    metrics = calculate_cardiac_metrics(
        sample['segmentations'][0].numpy(),  # ED
        sample['segmentations'][1].numpy(),  # ES
        sample['metadata']['spacing']
    )
    print(f"患者 {sample['patient_id']} LVEF: {metrics['lv_ef']:.1f}%")
```

## 🛠️ 文件结构

```
datalaoder/
├── acdc_dataset.py      # 核心数据集类
├── utils/               # 工具包
│   ├── __init__.py     # 工具包初始化
│   ├── transforms.py   # 数据变换函数
│   ├── analysis.py     # 数据分析工具
│   ├── metrics.py      # 评估指标计算
│   └── visualization.py # 可视化工具
├── tests/              # 测试文件
│   ├── __init__.py     # 测试包初始化
│   ├── test_all_methods.ipynb  # 完整测试notebook
│   └── test_unit.py    # 单元测试
├── example_usage.py    # 完整使用示例
├── __init__.py         # 包初始化
└── README.md          # 说明文档
```

## ⚠️ 注意事项

1. **数据路径**: 确保数据集路径正确，包含training和testing文件夹
2. **内存使用**: 大数据集建议关闭缓存(`cache_data=False`)
3. **多进程**: Windows用户可能需要设置`num_workers=0`
4. **GPU内存**: 4D数据加载时注意GPU内存限制

## 🔗 相关资源

- **ACDC数据集**: [官方网站](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
- **论文引用**: Bernard, O. et al. "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?" IEEE TMI 2018
- **许可证**: CC BY-NC-SA 4.0 (仅限非商业科研用途)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个数据加载器！

## 📄 许可证

本代码遵循MIT许可证。ACDC数据集本身遵循CC BY-NC-SA 4.0许可证。
