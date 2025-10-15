# 变更日志 (Changelog)

本项目采用语义化版本命名 (Semantic Versioning)。

## [1.0.1] - 2025-10-16

### Changed
- 🔧 修正文件夹拼写错误：将 `datalaoder` 重命名为 `dataloader`
- 📁 更新项目结构，保持所有文件内容和功能不变
- 🔄 同步Git配置，确保与远程仓库 https://github.com/Justinfungi/ICLR-Med-Compression.git 正确连接

### Fixed
- 🐛 修复文件夹命名不一致问题
- 🔗 确保Git上游分支正确设置和跟踪

### Security
- 🔒 验证远程仓库安全性，确保代码来源可信

---

## [1.0.0] - 2025-10-16

### Added
- 🫀 初始化医学图像压缩项目 (Medical Image Compression Project)
- 📦 添加完整的 ACDC 心脏 MRI 数据集处理工具链
- 🤖 集成多种深度学习模型 (SimpleSegModel, UNet3D, ResNet3D, CardiacNet)
- 📊 实现数据分析和可视化功能
- 🛠️ 添加数据预处理和增强功能
- 📈 集成心脏功能指标计算 (LVEF, RVEF, 心室容积等)
- 🎨 支持4D心跳周期动画生成
- 📚 创建详细的项目文档和使用指南

### Changed
- 🚀 优化项目结构，提高代码模块化程度
- 📖 更新README.md，提供完整的项目概述和使用说明

### Deprecated
- 📋 标记旧版本数据加载方式为已弃用

### Removed
- 🗑️ 删除不必要的PDF文件，精简仓库大小

### Fixed
- 🔧 修复数据加载和预处理中的潜在问题

### Security
- 🔒 确保数据集下载和处理的安全性
