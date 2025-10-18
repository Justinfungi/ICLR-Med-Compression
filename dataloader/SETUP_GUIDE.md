# ACDC数据加载器 - 安装与测试指南

## 📦 安装依赖

### 方法1: 使用requirements.txt (推荐)

```bash
cd /root/Documents/ICLR-Med/MedCompression/dataloader
pip install -r requirements.txt
```

### 方法2: 手动安装核心依赖

```bash
# 深度学习框架
pip install torch torchvision

# 科学计算
pip install numpy scipy pandas

# 医学图像处理
pip install SimpleITK

# 可视化
pip install matplotlib seaborn

# 进度条
pip install tqdm
```

### 可选依赖

```bash
# GIF动画生成
pip install imageio

# 数据下载
pip install huggingface-hub
```

## 🧪 测试安装

### 1. 验证导入

```bash
cd /root/Documents/ICLR-Med/MedCompression/dataloader
python3 -c "from acdc_dataset import ACDCDataset, ACDCDataModule; print('✅ Import successful')"
```

### 2. 运行完整示例

```bash
cd tests
python3 example_usage.py
```

**注意**: 运行示例前需要:
1. 下载ACDC数据集
2. 修改 `example_usage.py` 中的数据路径 (第76行)

```python
# 将此行修改为您的数据路径
data_root = "/path/to/your/acdc_dataset"
```

## 📁 数据集准备

### 下载ACDC数据集

1. 访问 [ACDC Challenge官网](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
2. 注册并下载数据集
3. 解压到指定目录

### 预期目录结构

```
acdc_dataset/
├── training/
│   ├── patient001/
│   │   ├── Info.cfg
│   │   ├── patient001_4d.nii.gz
│   │   ├── patient001_frame01.nii.gz
│   │   ├── patient001_frame01_gt.nii.gz
│   │   └── ...
│   ├── patient002/
│   └── ...
└── testing/
    └── ...
```

## ✅ 快速验证

运行以下代码验证数据加载器工作正常:

```python
from acdc_dataset import ACDCDataset

# 创建数据集 (使用您的数据路径)
dataset = ACDCDataset(
    data_root="/path/to/acdc_dataset",
    split='training',
    mode='3d_keyframes',
    load_segmentation=True
)

# 获取第一个样本
sample = dataset[0]
print(f"✅ 数据集大小: {len(dataset)}例患者")
print(f"✅ 样本形状: {sample['images'].shape}")
print(f"✅ 疾病类型: {sample['patient_info']['Group']}")
```

## 🔧 常见问题

### Q1: ImportError: No module named 'SimpleITK'
**解决**: 安装SimpleITK
```bash
pip install SimpleITK
```

### Q2: FileNotFoundError: 数据集路径不存在
**解决**: 检查并修正数据路径
```python
data_root = Path("/your/correct/path/to/acdc_dataset")
assert data_root.exists(), f"数据路径不存在: {data_root}"
```

### Q3: 内存不足
**解决**: 关闭数据缓存
```python
dataset = ACDCDataset(
    cache_data=False  # 关闭内存缓存
)
```

### Q4: Windows多进程报错
**解决**: 设置单进程加载
```python
data_module = ACDCDataModule(
    num_workers=0  # 使用单进程
)
```

## 📚 文档资源

- **完整README**: `README.md` (英文详细文档)
- **中文简明指南**: `README_CN.md`
- **使用示例**: `tests/example_usage.py`
- **数据集分析**: 原README.md中的详细说明

## 💡 下一步

安装完成后,建议:

1. ✅ 运行 `tests/example_usage.py` 熟悉功能
2. ✅ 查看 `README_CN.md` 了解核心用法
3. ✅ 参考 `README.md` 深入理解数据集
4. ✅ 根据任务需求定制数据加载配置

祝您使用愉快! 🚀
