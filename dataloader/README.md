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

## 📊 数据来源与处理流程

### 🏥 ACDC数据集结构

ACDC (Automated Cardiac Diagnosis Challenge) 是一个标准的心脏MRI数据集，包含100例训练数据和50例测试数据。

#### 📁 原始数据文件结构
```
acdc_dataset/
├── training/
│   ├── patient001/
│   │   ├── Info.cfg                      # 患者元数据
│   │   ├── patient001_4d.nii.gz         # 完整4D心跳序列 [T,Z,H,W]
│   │   ├── patient001_frame01.nii.gz    # ED时相图像 [Z,H,W]
│   │   ├── patient001_frame12.nii.gz    # ES时相图像 [Z,H,W]
│   │   ├── patient001_frame01_gt.nii.gz # ED分割标注 [Z,H,W] ⭐
│   │   ├── patient001_frame12_gt.nii.gz # ES分割标注 [Z,H,W] ⭐
│   │   └── MANDATORY_CITATION.md
│   └── ...
└── testing/
    └── ...
```

#### 📊 **详细数据形状与空间分辨率分析**

基于实际Patient数据的完整统计：

| 文件类型 | 典型形状 | 数据类型 | 体素间距范围 | 体素体积 | 文件大小 |
|---------|----------|----------|-------------|----------|----------|
| **4D序列** | `(28-30, 10, 216-256, 216-256)` | `int16` | `1.37-1.56mm × 1.37-1.56mm × 10mm` | `18.7-24.4 mm³` | `15-17 MB` |
| **ED/ES图像** | `(10, 216-256, 216-256)` | `int16` | `1.37-1.56mm × 1.37-1.56mm × 10mm` | `18.7-24.4 mm³` | `0.6 MB` |
| **分割标注** | `(10, 216-256, 216-256)` | `uint8` | `1.37-1.56mm × 1.37-1.56mm × 10mm` | `18.7-24.4 mm³` | `0.03 MB` |

##### 🔍 **具体患者示例对比**

```python
# Patient001 (DCM) - 标准分辨率
patient001_4d.shape:    (30, 10, 256, 216)  # 30帧 × 10切片 × 256×216像素
patient001_spacing:     (1.5625, 1.5625, 10.0) mm
patient001_voxel_vol:   24.41 mm³

# Patient002 (DCM) - 高分辨率
patient002_4d.shape:    (30, 10, 256, 232)  # 30帧 × 10切片 × 256×232像素  
patient002_spacing:     (1.3671875, 1.3671875, 10.0) mm
patient002_voxel_vol:   18.69 mm³

# Patient005 (DCM) - 中等分辨率
patient005_4d.shape:    (30, 10, 216, 256)  # 30帧 × 10切片 × 216×256像素
patient005_spacing:     (1.40625, 1.40625, 10.0) mm  
patient005_voxel_vol:   19.78 mm³
```

##### 📐 **维度说明**

```python
# 4D序列: [时间, 切片, 高度, 宽度]
4D_shape = (T, Z, H, W)
T: 28-30 帧     # 心跳周期帧数，取决于心率
Z: 10 切片      # 短轴切片数，覆盖心脏从底部到顶点
H: 216-256 像素 # 图像高度，前后方向
W: 216-256 像素 # 图像宽度，左右方向

# 3D关键帧: [切片, 高度, 宽度]  
3D_shape = (Z, H, W)
Z: 10 切片      # 与4D相同的切片覆盖
H: 216-256 像素 # 与4D相同的空间分辨率
W: 216-256 像素 # 与4D相同的空间分辨率
```

#### 📋 患者元数据 (Info.cfg)
```
ED: 1          # 舒张末期帧号
ES: 12         # 收缩末期帧号
Group: DCM     # 疾病类型 (DCM/HCM/MINF/NOR/RV)
Height: 184.0  # 身高 (cm)
Weight: 95.0   # 体重 (kg)
NbFrame: 30    # 总帧数
```

#### 🏷️ **分割标签定义与体素分布**

```python
# 每个体素的标签值 (来自 *_gt.nii.gz 文件)
0: 背景 (Background)        # 心脏外部组织和空气
1: 右心室腔 (RV Cavity)     # 右心室血池 - RVEF计算
2: 左心室心肌 (LV Myocardium) # 左心室肌肉 - 心肌质量计算
3: 左心室腔 (LV Cavity)     # 左心室血池 - LVEF计算 ⭐
```

##### 📊 **实际分割分布统计 (Patient002示例)**

| 时相 | 标签 | 解剖结构 | 体素数量 | 占比 | 实际体积 |
|------|------|----------|----------|------|----------|
| **ED帧** | 0 | 背景 | 566,068 | 95.2% | - |
|  | 1 | 右心室腔 | 5,052 | 0.85% | **94.4 ml** |
|  | 2 | 左心室心肌 | 8,583 | 1.44% | **160.4 ml** |
|  | 3 | 左心室腔 | 14,217 | 2.39% | **265.7 ml** ⭐ |
| **ES帧** | 0 | 背景 | 572,002 | 96.2% | - |
|  | 1 | 右心室腔 | 1,542 | 0.26% | **28.8 ml** |
|  | 2 | 左心室心肌 | 10,302 | 1.73% | **192.5 ml** |
|  | 3 | 左心室腔 | 10,074 | 1.69% | **188.3 ml** ⭐ |

```python
# 计算公式 (以Patient002为例)
体素间距 = (1.3671875, 1.3671875, 10.0) mm
体素体积 = 1.3671875 × 1.3671875 × 10.0 = 18.69 mm³

# 左心室射血分数计算
LVEDV = 14,217 × 18.69 / 1000 = 265.7 ml  # 舒张末期容积
LVESV = 10,074 × 18.69 / 1000 = 188.3 ml  # 收缩末期容积
LVEF = (LVEDV - LVESV) / LVEDV × 100% = 29.1%  # 典型DCM低射血分数
```

### 🔄 数据处理流程

#### 1. **NIfTI文件加载**
```python
def _load_nifti(self, filepath: Path) -> Tuple[np.ndarray, Dict]:
    """
    使用SimpleITK加载医学图像，提取：
    - 3D图像数组: shape (Z, H, W)
    - 元数据: spacing, origin, direction等
    """
    image = sitk.ReadImage(str(filepath))
    image_array = sitk.GetArrayFromImage(image)
    
    metadata = {
        'spacing': image.GetSpacing(),    # ⭐ 体素间距 (x,y,z) mm
        'origin': image.GetOrigin(),      # 图像原点
        'direction': image.GetDirection() # 方向矩阵
    }
    return image_array, metadata
```

#### 2. **心脏功能指标计算**
```python
def calculate_cardiac_metrics(ed_seg, es_seg, spacing):
    """
    基于分割标注计算心脏功能指标
    
    核心公式：
    - 体积 = 体素数量 × 体素间距乘积 / 1000 (转为ml)
    - LVEF = (LVEDV - LVESV) / LVEDV × 100%
    - RVEF = (RVEDV - RVESV) / RVEDV × 100%
    """
    # 体素体积 (mm³)
    voxel_volume = np.prod(spacing)
    
    # 计算各结构体积
    def calculate_volume(seg_map, label):
        voxel_count = np.sum(seg_map == label)
        return voxel_count * voxel_volume / 1000  # ml
    
    # ED/ES时相各结构体积
    ed_lv_cavity = calculate_volume(ed_seg, 3)    # 左心室腔-ED
    es_lv_cavity = calculate_volume(es_seg, 3)    # 左心室腔-ES
    ed_rv_cavity = calculate_volume(ed_seg, 1)    # 右心室腔-ED
    es_rv_cavity = calculate_volume(es_seg, 1)    # 右心室腔-ES
    
    return {
        'lv_edv': ed_lv_cavity,                           # 左心室舒张末期容积
        'lv_esv': es_lv_cavity,                           # 左心室收缩末期容积
        'lv_ef': (ed_lv_cavity - es_lv_cavity) / ed_lv_cavity * 100,  # 射血分数
        'rv_ef': (ed_rv_cavity - es_rv_cavity) / ed_rv_cavity * 100,
        # ... 更多指标
    }
```

#### 3. **数据验证示例**
```python
# 实际计算示例 (patient001, DCM患者)
spacing = (1.5625, 1.5625, 10.0)  # 从NIfTI文件头部获取
ed_seg = load("patient001_frame01_gt.nii.gz")  # 专家标注的分割
es_seg = load("patient001_frame12_gt.nii.gz")

metrics = calculate_cardiac_metrics(ed_seg, es_seg, spacing)
# 结果:
# LVEDV: 295.5 ml (正常范围扩大，符合DCM特征)
# LVESV: 225.6 ml  
# LVEF: 23.7% (严重降低，确认DCM诊断)
# 心功能评估: 重度减低 ✅
```

### 🎯 疾病类型与临床意义

| 疾病代码 | 疾病名称 | 典型LVEF范围 | 主要特征 |
|---------|----------|-------------|---------|
| NOR | 正常心脏 | ≥55% | 心功能正常 |
| DCM | 扩张性心肌病 | <40% | 心室扩大，收缩功能减退 |
| HCM | 肥厚性心肌病 | ≥55% | 心肌肥厚，舒张功能障碍 |
| MINF | 心肌梗死 | 变异大 | 局部室壁运动异常 |
| RV | 右心室异常 | 正常 | 右心室病变 |

### 🔍 数据质量保证

#### ✅ **医学权威性**
- 数据来源: 法国里昂第一大学CREATIS实验室
- 标注质量: 专业放射科医生手工标注
- 验证标准: 临床诊断与图像标注一致性验证

#### ✅ **技术准确性**
- 格式标准: 符合NIfTI医学图像标准
- 元数据完整: 包含完整的空间信息和患者信息
- 计算验证: 心脏功能指标计算符合临床标准

#### 📖 **引用要求**
```
O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al.
"Deep Learning Techniques for Automatic MRI Cardiac Multi-structures 
Segmentation and Diagnosis: Is the Problem Solved?" 
IEEE Transactions on Medical Imaging, vol. 37, no. 11, pp. 2514-2525, 2018.
```

### 💡 技术实现亮点

#### 🔄 **多模式数据加载**
```python
# 支持的加载模式与输出形状
modes = {
    '3d_keyframes': "加载ED和ES关键帧 [2, Z, H, W]",
    '4d_sequence': "加载完整心跳序列 [T, Z, H, W]", 
    'ed_only': "仅加载舒张末期 [Z, H, W]",
    'es_only': "仅加载收缩末期 [Z, H, W]"
}

# 4D序列加载的实际实现
def load_4d_sequence(patient_id):
    """加载完整的4D心跳序列"""
    file_path = f"{data_root}/{split}/{patient_id}/{patient_id}_4d.nii.gz"
    
    # 读取4D NIfTI文件
    image_4d = sitk.ReadImage(file_path)
    array_4d = sitk.GetArrayFromImage(image_4d)  
    
    # 输出形状: (时间帧, Z切片, H高度, W宽度)
    # 实例: (30, 10, 256, 216) 表示30个时间帧
    
    metadata = {
        'spacing': image_4d.GetSpacing(),      # (X, Y, Z, T)
        'temporal_resolution': spacing[3],      # 时间分辨率 (通常为1.0)
        'cardiac_phases': array_4d.shape[0],   # 心跳周期总帧数
    }
    
    return array_4d, metadata

# 自动识别ED/ES帧号
info = self._parse_info_file(info_file)
ed_frame = info.get('ED', 1)     # 从Info.cfg读取 (1-based)
es_frame = info.get('ES', 14)    # 自动匹配对应文件 (1-based)

# 从4D序列中提取关键帧
def extract_keyframes_from_4d(array_4d, ed_frame, es_frame):
    """从4D序列提取ED/ES关键帧"""
    # 转换为0-based索引
    ed_idx = ed_frame - 1
    es_idx = es_frame - 1
    
    ed_image = array_4d[ed_idx]  # [Z, H, W]
    es_image = array_4d[es_idx]  # [Z, H, W]
    
    return np.stack([ed_image, es_image])  # [2, Z, H, W]
```

#### 🎯 **智能预处理流程**
```python
def _preprocess_data(self, data):
    """完整的数据预处理管道"""
    
    # 1. 重采样到统一分辨率
    if self.target_spacing is not None:
        original_spacing = data['metadata']['spacing']
        for key in ['image', 'images', 'segmentation', 'segmentations']:
            if key in data and data[key] is not None:
                data[key] = self._resample_image(
                    data[key], original_spacing, self.target_spacing
                )
    
    # 2. 强度归一化
    if self.normalize:
        for key in ['image', 'images']:
            if key in data and data[key] is not None:
                data[key] = self._normalize_intensity(data[key])
    
    return data
```

#### 📊 **数据流追踪示例**
```python
# 完整的数据处理链路
patient_id = "patient001"

# Step 1: 文件路径解析
files = {
    'ed_image': "patient001_frame01.nii.gz",
    'es_image': "patient001_frame12.nii.gz", 
    'ed_seg': "patient001_frame01_gt.nii.gz",
    'es_seg': "patient001_frame12_gt.nii.gz"
}

# Step 2: NIfTI加载 + 元数据提取
ed_img, metadata = self._load_nifti(files['ed_image'])
# ed_img.shape: (10, 256, 216) - 原始图像
# metadata['spacing']: (1.5625, 1.5625, 10.0) - 体素间距

# Step 3: 分割标注加载
ed_seg, _ = self._load_nifti(files['ed_seg'])
# ed_seg中: 标签3的体素 = 左心室腔

# Step 4: 预处理
if self.target_spacing:
    ed_img = self._resample_image(ed_img, metadata['spacing'], self.target_spacing)
    ed_seg = self._resample_image(ed_seg, metadata['spacing'], self.target_spacing)

# Step 5: 堆叠为批次
data = {
    'images': np.stack([ed_img, es_img]),           # [2, Z, H, W]
    'segmentations': np.stack([ed_seg, es_seg]),    # [2, Z, H, W]
    'metadata': metadata,
    'patient_info': {'Group': 'DCM', 'Height': 184.0, ...}
}

# Step 6: 转换为PyTorch张量
for key in ['images', 'segmentations']:
    data[key] = torch.from_numpy(data[key]).float()
    if 'segmentation' in key:
        data[key] = data[key].long()  # 标签数据用long类型
```

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
