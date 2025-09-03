# 🫀 Medical Image Compression Project

## 📖 项目概述

这是一个专注于医学图像压缩的研究项目，提供了完整的 ACDC 心脏 MRI 数据集处理工具链和深度学习模型。

## ✨ 核心功能

### 🔧 数据处理
- **多模式数据加载**: 支持 3D 关键帧、4D 序列、ED/ES 单帧模式
- **智能预处理**: 自动重采样、强度归一化、数据增强
- **批量处理**: 高效的 DataLoader 和多进程支持

### 📊 数据分析
- **统计分析**: 完整的数据集统计和分布分析
- **心脏指标**: LVEF、RVEF、心室容积、心肌质量计算
- **质量评估**: Dice、Hausdorff 距离、体积相似性

### 🎨 可视化
- **心脏时相显示**: ED/ES 对比可视化
- **动画生成**: 4D 心跳周期 GIF 动画
- **统计图表**: 疾病分布、患者统计图表
- **分割叠加**: 图像与分割结果叠加显示

### 🤖 深度学习
- **多种模型**: SimpleSegModel、UNet3D、ResNet3D、CardiacNet
- **训练管道**: 完整的训练、验证、测试流程
- **模型对比**: 不同架构的性能对比

## 🚀 快速开始

### 1. 环境配置
```bash
# 克隆仓库
git clone https://github.com/Justinfungi/ICLR-Med-Compression.git
cd ICLR-Med-Compression

# 安装依赖
pip install torch torchvision numpy matplotlib SimpleITK imageio
```

### 2. 数据准备
```bash
# 下载 ACDC 数据集
python scripts/acdc_download.py

# 数据集结构
acdc_dataset/
├── training/
│   ├── patient001/
│   │   ├── patient001_frame01.nii.gz
│   │   ├── patient001_frame01_gt.nii.gz
│   │   └── ...
│   └── ...
└── testing/
    └── ...
```

### 3. 运行示例
```bash
cd datalaoder/tests
python example_usage.py
```

### 4. 交互式菜单
```
🫀 ACDC数据集加载器 - 功能选择菜单
============================================================
📋 可用功能:
  1️⃣  基本使用示例
  2️⃣  数据模块使用示例
  3️⃣  数据变换示例
  4️⃣  4D时序数据示例
  5️⃣  心脏功能指标计算示例
  6️⃣  数据集分析示例
  7️⃣  简单训练示例
  8️⃣  🌟 模型选择和完整功能测试
  ──────────────────────────────────────
  🔧 D  调试基本功能 (快速问题诊断)
  🔄 A  运行所有示例 (推荐新用户)
  💡 H  显示详细帮助信息
  ❌ Q  退出程序
============================================================
```

## 📁 项目结构

```
MedCompression/
├── datalaoder/                 # 📦 核心数据加载模块
│   ├── acdc_dataset.py        # ACDC数据集类
│   ├── utils/                 # 工具函数
│   │   ├── transforms.py      # 数据变换
│   │   ├── analysis.py        # 数据分析
│   │   ├── metrics.py         # 评估指标
│   │   └── visualization.py   # 可视化工具
│   └── tests/
│       └── example_usage.py   # 🎯 交互式示例程序
├── docs/                      # 📚 文档和论文
├── scripts/                   # 🛠️ 工具脚本
└── README.md                  # 📖 项目说明
```

## 🎯 主要特性

### 🔇 静默模式
- 自动抑制所有 SimpleITK 和 matplotlib 警告
- 图像直接保存，不弹出显示窗口
- 干净清爽的命令行输出

### 📁 自动输出管理
- 时间戳命名的输出目录
- 自动保存所有结果（图像、动画、报告、日志）
- 完整的实验记录和可重现性

### 🎨 丰富的可视化
- 心脏时相对比图
- 4D 心跳动画 GIF
- 疾病分布统计图表
- 分割结果叠加显示

## 🧠 支持的模型

### 分割模型
- **SimpleSegModel**: 轻量级 3D CNN
- **UNet3D**: 经典的 3D U-Net 架构
- **CardiacNet**: 心脏专用多任务网络

### 分类模型
- **ResNet3DClassifier**: 3D ResNet 疾病分类

## 📊 数据集信息

**ACDC (Automated Cardiac Diagnosis Challenge)**
- 🏥 **来源**: 法国里昂第一大学 CREATIS 实验室
- 📈 **规模**: 100 例训练 + 50 例测试
- 🔬 **模态**: 短轴心脏 MRI (cine-MRI)
- 🎯 **任务**: 心脏分割 + 疾病分类
- 📋 **疾病类型**: NOR, DCM, HCM, MINF, RV (5类)

## 🔧 系统要求

- Python 3.8+
- PyTorch 1.8+
- GPU 推荐（用于模型训练）
- 磁盘空间: ~2GB（数据集）

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📞 联系方式

- **作者**: Justinfungi
- **项目主页**: https://github.com/Justinfungi/ICLR-Med-Compression
- **问题反馈**: https://github.com/Justinfungi/ICLR-Med-Compression/issues

---

🎉 **开始探索医学图像压缩的奇妙世界吧！**
