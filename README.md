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

## 🚀 Git 操作指南

本项目使用 Git 进行版本控制，并包含了 `1d-tokenizer` 作为 Git submodule。以下是常见的 Git 操作：

### 1. 克隆仓库 (包含 Submodule)

如果你是第一次克隆本项目，请使用 `--recurse-submodules` 选项来同时克隆主仓库和所有 Submodule：

```bash
git clone --recurse-submodules https://github.com/Justinfungi/ICLR-Med-Compression.git
cd ICLR-Med-Compression
```

如果你已经克隆了主仓库但没有克隆 Submodule，或者 Submodule 目录是空的，可以手动初始化和更新 Submodule：

```bash
cd ICLR-Med-Compression
git submodule update --init --recursive
```

### 2. 获取远程更新 (Fetch)

`git fetch` 命令用于从远程仓库下载最新的提交、分支和标签，但不会自动合并或修改你本地的工作目录。这可以让你查看远程仓库的最新状态，而不会影响你当前的工作：

```bash
git fetch origin
```

要查看 `1d-tokenizer` Submodule 的更新：

```bash
cd 1d-tokenizer
git fetch origin
```

### 3. 拉取并合并更新 (Pull)

`git pull` 命令是 `git fetch` 和 `git merge` 的组合，它会从远程仓库获取更新并自动合并到你当前的分支：

```bash
# 拉取主仓库的更新并合并到当前分支
git pull origin <your-current-branch>
# 例如：git pull origin feature/20251019-mvp1
```

要拉取 `1d-tokenizer` Submodule 的更新：

```bash
cd 1d-tokenizer
git pull origin main # 假设 submodule 的主分支是 main
cd .. # 返回主仓库目录
# 提交 submodule 引用更新
git add 1d-tokenizer
git commit -m "Update 1d-tokenizer submodule to latest upstream"
```

### 4. 更新所有 Submodule

如果你有多个 Submodule，并希望将它们全部更新到主仓库中记录的最新提交，可以从主仓库的根目录运行：

```bash
git submodule update --remote
```
**注意**: 执行此命令后，如果 `submodule` 有更新，主仓库会检测到 `submodule` 引用的变化，你需要 `git add 1d-tokenizer` (或其他 submodule 路径) 并 `git commit -m "Update submodule references"` 来记录这个变化。

### 5. 推送本地更改 (Push)

当你完成了本地的更改（包括主仓库和 Submodule 的更改）并提交后，可以使用 `git push` 将它们上传到远程仓库：

```bash
# 推送当前分支到远程仓库
git push origin <your-current-branch>
# 例如：git push origin feature/20251019-mvp1
```

你也可以使用项目根目录下的 `push_to_github.sh` 脚本来简化推送过程，它会自动添加、提交并推送所有更改：

```bash
./push_to_github.sh
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
├── 1d-tokenizer/              # ⚙️ TiTok tokenization submodule
└── README.md                  # 📖 项目说明
```