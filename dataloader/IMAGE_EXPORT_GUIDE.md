# 📸 图像导出功能说明

## ✨ 新增功能

修改后的 `example_basic_usage()` 函数现在会**分别保存**每个图像,使用**描述性文件名**。

## 📁 导出的文件

运行示例后,会在输出目录中生成以下图像文件:

### 1. 组合图像 (Combined View)
```
patient001_DCM_combined_slice5.png
```
- 包含: ED图像 + ED分割 + ES图像 + ES分割
- 用途: 快速浏览所有时相

### 2. ED时相图像 (End-Diastolic Images)
```
patient001_DCM_ED_image_slice5.png          # ED原始图像
patient001_DCM_ED_segmentation_slice5.png   # ED分割标注
patient001_DCM_ED_overlay_slice5.png        # ED叠加图(图像+分割)
```

### 3. ES时相图像 (End-Systolic Images)
```
patient001_DCM_ES_image_slice5.png          # ES原始图像
patient001_DCM_ES_segmentation_slice5.png   # ES分割标注
patient001_DCM_ES_overlay_slice5.png        # ES叠加图(图像+分割)
```

## 📝 文件命名规则

```
格式: {患者ID}_{疾病类型}_{时相}_{类型}_slice{切片号}.png

示例:
patient001_DCM_ED_image_slice5.png
│          │   │  │     │
│          │   │  │     └─ 切片编号
│          │   │  └─ 图像类型 (image/segmentation/overlay/combined)
│          │   └─ 心脏时相 (ED=舒张末期, ES=收缩末期)
│          └─ 疾病类型 (DCM/HCM/MINF/NOR/ARV)
└─ 患者ID
```

## 🎨 图像类型说明

| 类型 | 文件名 | 说明 | 用途 |
|------|--------|------|------|
| **image** | `*_image_*.png` | 原始MRI图像 (灰度) | 查看心脏结构 |
| **segmentation** | `*_segmentation_*.png` | 分割标注 (彩色) | 查看标注质量 |
| **overlay** | `*_overlay_*.png` | 图像+分割叠加 | 验证分割准确性 |
| **combined** | `*_combined_*.png` | 所有视图组合 | 全面对比 |

## 🎯 典型输出示例

运行 `example_basic_usage()` 后,会看到:

```
💾 组合图像已保存: output/.../patient001_DCM_combined_slice5.png
💾 ED图像已保存: output/.../patient001_DCM_ED_image_slice5.png
💾 ES图像已保存: output/.../patient001_DCM_ES_image_slice5.png
💾 ED分割已保存: output/.../patient001_DCM_ED_segmentation_slice5.png
💾 ES分割已保存: output/.../patient001_DCM_ES_segmentation_slice5.png
💾 ED叠加图已保存: output/.../patient001_DCM_ED_overlay_slice5.png
💾 ES叠加图已保存: output/.../patient001_DCM_ES_overlay_slice5.png

✨ 共保存了 7 张图像到: output/acdc_results_YYYYMMDD_HHMMSS/basic_usage
```

## 📊 图像详细说明

### ED时相 (End-Diastolic - 舒张末期)
- **时间点**: 心脏充盈最大时
- **特征**: 心室容积最大,心肌相对较薄
- **临床意义**: 用于计算舒张末期容积 (EDV)

### ES时相 (End-Systolic - 收缩末期)
- **时间点**: 心脏收缩最大时
- **特征**: 心室容积最小,心肌相对较厚
- **临床意义**: 用于计算收缩末期容积 (ESV)

### 叠加图 (Overlay)
- **显示**: 原始图像 (灰度) + 分割掩码 (彩色半透明)
- **颜色**: Jet色图,不同标签用不同颜色
- **用途**: 直观验证分割是否准确覆盖心脏结构

## 🔍 分割标签颜色说明

在分割图和叠加图中:

| 标签值 | 结构 | 颜色 (Viridis/Jet) |
|-------|------|-------------------|
| 0 | 背景 | 深蓝/紫色 |
| 1 | 右心室腔 (RV) | 蓝绿色 |
| 2 | 左心室心肌 (LV-Myo) | 黄绿色 |
| 3 | 左心室腔 (LV) | 黄色/红色 |

## 💡 使用技巧

### 1. 批量导出多个患者
```python
for i in range(10):  # 导出前10个患者
    sample = dataset[i]
    # 调用可视化代码...
```

### 2. 导出所有切片
```python
for slice_idx in range(images.shape[1]):
    # 为每个切片保存图像...
```

### 3. 自定义文件名
```python
custom_name = f"{patient_id}_custom_description_slice{slice_idx}.png"
plt.savefig(output_dir / custom_name, dpi=300, bbox_inches='tight')
```

## 📂 输出目录结构

```
output/acdc_results_20251018_200922/basic_usage/
├── patient001_DCM_combined_slice5.png
├── patient001_DCM_ED_image_slice5.png
├── patient001_DCM_ED_segmentation_slice5.png
├── patient001_DCM_ED_overlay_slice5.png
├── patient001_DCM_ES_image_slice5.png
├── patient001_DCM_ES_segmentation_slice5.png
└── patient001_DCM_ES_overlay_slice5.png
```

## 🎬 运行示例

```bash
cd /root/Documents/ICLR-Med/MedCompression/dataloader/tests
python example_usage.py

# 选择选项 1 (基本使用示例)
# 图像将自动保存到 output/ 目录
```

## 📸 图像质量

- **分辨率**: 300 DPI (高质量,适合论文发表)
- **格式**: PNG (无损压缩)
- **尺寸**:
  - 单图: 8×8英寸 (2400×2400像素)
  - 组合图: 16×4英寸 或 4×16英寸

## ⚙️ 自定义配置

修改代码中的参数以调整输出:

```python
# 调整DPI (分辨率)
plt.savefig(path, dpi=150)  # 降低以减小文件大小

# 调整图像尺寸
fig, ax = plt.subplots(1, 1, figsize=(10, 10))  # 更大的图像

# 更改颜色映射
ax.imshow(image, cmap='hot')  # 热度图
ax.imshow(seg, cmap='tab20')  # 更多颜色
```

## 🔗 相关功能

- **完整示例**: `tests/example_usage.py`
- **可视化工具**: `utils/visualization.py`
- **数据加载**: `acdc_dataset.py`

---

**更新时间**: 2025-10-18
**版本**: v2.0 - 增强图像导出功能
