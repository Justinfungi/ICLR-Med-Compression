# 🖼️ ACDC数据集帧导出工具

## 📖 简介

此工具用于批量导出ACDC心脏MRI数据集的所有帧为PNG图像格式，便于后续的图像处理、分析和可视化任务。

## ✨ 功能特性

- **多模式支持**: 支持4D时序数据、ED/ES关键帧、单时相数据的完整导出
- **智能命名**: 标准化的文件命名约定，便于数据管理和后续处理
- **批量处理**: 高效的批量导出，支持进度显示和错误处理
- **高质量输出**: 300 DPI PNG图像，适合论文发表和专业分析
- **灵活配置**: 支持多种导出模式和目录结构选项

## 🚀 快速开始

### 基本用法

```bash
# 导出4D时序数据的全部帧（推荐）
python export_all_frames.py --mode 4d_sequence

# 导出ED/ES关键帧
python export_all_frames.py --mode 3d_keyframes

# 只导出ED时相
python export_all_frames.py --mode ed_only
```

### 高级用法

```bash
# 导出前5个患者的所有帧
python export_all_frames.py --mode 4d_sequence --end_idx 5

# 导出特定范围的患者
python export_all_frames.py --mode 4d_sequence --start_idx 10 --end_idx 20

# 自定义输出目录
python export_all_frames.py --target_dir ../my_images --mode 4d_sequence

# 不创建子目录（所有图像保存到同一目录）
python export_all_frames.py --no_subdirs --mode 4d_sequence
```

## 📋 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | `../acdc_dataset` | ACDC数据集根目录 |
| `--target_dir` | str | `./acdc_img_datasets` | 目标保存目录 |
| `--mode` | str | `4d_sequence` | 数据加载模式 |
| `--start_idx` | int | `0` | 开始患者索引 |
| `--end_idx` | int | `None` | 结束患者索引 (None表示全部) |
| `--no_subdirs` | flag | - | 不创建患者子目录 |
| `--verbose` | flag | - | 显示详细信息 |

### 数据加载模式

- **`4d_sequence`**: 导出完整的心跳周期4D数据（推荐）
- **`3d_keyframes`**: 导出ED和ES关键帧的所有切片
- **`ed_only`**: 只导出ED（舒张末期）时相的所有切片
- **`es_only`**: 只导出ES（收缩末期）时相的所有切片

## 📁 输出结构

### 目录结构

```
acdc_img_datasets/
├── export_summary.md              # 导出摘要报告
├── patient001/                    # 患者1目录
│   ├── patient001_frame001.png
│   ├── patient001_frame002.png
│   └── ...
├── patient002/                    # 患者2目录
│   └── ...
└── ...
```

### 文件命名规则

- **4D序列模式**: `{patient_id}_frame{frame_idx:03d}.png`
  - 示例: `patient001_frame001.png`, `patient001_frame030.png`

- **关键帧模式**: `{patient_id}_{phase}_frame{slice_idx:03d}.png`
  - 示例: `patient001_ED_frame001.png`, `patient001_ES_frame010.png`

## 📊 输出示例

运行完成后，您将看到类似以下的输出：

```
🖼️ ACDC数据集帧导出工具
==================================================
📁 数据目录: ../acdc_dataset
🎯 目标目录: acdc_img_datasets
📊 导出模式: 4d_sequence
👥 患者范围: 0 - 全部
📂 子目录: 是

🚀 开始批量导出帧
👥 处理患者: 1 - 100 (共 100 例)
📊 模式: 4d_sequence
------------------------------------------------------------
[██████████████████████████████] 100.0% 100/100 | ...

============================================================
🎉 导出完成!
👥 总患者数: 100
✅ 成功患者: 100
🖼️ 总帧数: 3000
⏱️ 总耗时: 15.5分钟
📊 平均每患者耗时: 9.3秒
🚀 处理速度: 3.2帧/秒
============================================================
```

## 📈 性能统计

### 典型性能数据

- **处理速度**: 3-6帧/秒（取决于硬件配置）
- **存储空间**: 每个患者约30帧 × ~50KB = ~1.5MB
- **总数据集**: 100患者 × 1.5MB ≈ 150MB

### 硬件要求

- **内存**: 至少4GB可用内存
- **存储**: 至少200MB可用磁盘空间
- **CPU**: 多核CPU推荐（用于并行处理）

## 🔧 故障排除

### 常见问题

1. **数据集路径错误**
   ```
   错误: 数据集目录不存在
   解决: 检查 --data_root 参数指向正确的ACDC数据集目录
   ```

2. **依赖包缺失**
   ```
   错误: No module named 'SimpleITK'
   解决: 运行 `pip install -r requirements.txt`
   ```

3. **磁盘空间不足**
   ```
   错误: 磁盘空间不足
   解决: 清理磁盘空间或指定更小的患者范围
   ```

4. **权限问题**
   ```
   错误: 无法创建目录
   解决: 检查目标目录的写入权限
   ```

### 调试模式

使用 `--verbose` 参数获取详细的处理信息：

```bash
python export_all_frames.py --mode 4d_sequence --end_idx 2 --verbose
```

## 📋 依赖要求

确保安装了以下依赖包：

```
numpy>=1.19.0
scipy>=1.5.0
SimpleITK>=2.0.0
pandas>=1.2.0
matplotlib>=3.3.0
```

安装命令：
```bash
pip install -r requirements.txt
```

## 🎯 使用建议

### 推荐工作流程

1. **小规模测试**: 先用 `--end_idx 2` 测试前几个患者
2. **全量导出**: 确认无误后运行完整导出
3. **数据验证**: 检查导出的图像质量和命名正确性
4. **后续处理**: 使用导出的PNG图像进行模型训练或分析

### 数据集说明

- **ACDC数据集**: 包含100例患者的4D心脏MRI数据
- **时相信息**: 每个心跳周期约25-35帧
- **空间分辨率**: 256×256像素，10-12个切片层
- **疾病类型**: 正常、肥厚性心肌病、扩张性心肌病等

## 📞 技术支持

如果遇到问题，请：

1. 检查错误信息和日志输出
2. 验证数据集完整性和路径正确性
3. 确认所有依赖包已正确安装
4. 查看 `export_summary.md` 文件获取详细统计信息

---

**更新时间**: 2025-10-18
**版本**: v1.0 - 初始发布



