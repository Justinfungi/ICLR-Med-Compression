# 变更日志

## [1.3.0] - 2025-10-18

### Added
- 新增命令行参数支持：添加`--extract_mode`参数选择提取模式
- 灵活的帧提取选项：支持`keyframe`（关键帧）和`all_frames`（所有帧）两种模式
- 增强参数化配置：支持自定义源目录和输出目录路径
- 改进用户界面：更新帮助文档和使用示例

### Changed
- 重构ACDCConverter类：添加extract_mode参数支持条件化处理逻辑
- 优化关键帧选择：使用心脏周期中间帧作为代表性关键帧（end-diastolic相位）
- 改进文件名规范：关键帧模式使用`_keyframe.png`后缀区分

### Details
- **提取模式**: `keyframe`模式提取单个代表性帧（心脏舒张末期），`all_frames`模式提取完整心脏周期
- **向后兼容**: 默认模式为`keyframe`，保持与之前版本的行为一致
- **参数验证**: 添加choices限制确保只接受有效参数值

## [1.2.9] - 2025-10-18

### Changed
- 增强ACDC数据集转换器：支持4D心脏MRI数据处理，提取所有时间帧而不仅是关键帧
- 更新convert_acdc.py脚本：添加对4D数据的完整支持，循环处理所有时间帧维度
- 优化图像命名规范：4D数据输出文件名包含slice和frame索引，便于后续处理

### Details
- **4D数据支持**: 心脏MRI数据通常为4D格式[x,y,slice,time]，新增对时间维度的完整提取
- **文件名格式**: 4D图像保存为`{stem}_slice_{slice_idx:03d}_frame_{frame_idx:03d}.png`
- **向后兼容**: 保持对2D和3D数据的原有处理逻辑不变

## [1.2.8] - 2025-10-18

### Fixed
- 修正H800 GPU集群SBATCH脚本配置：改用q-hgpu-batch分区，指定gpucluster-g4可用节点
- 优化训练参数：批大小16，训练轮数300，利用7天时间限制
- 解决"Requested node configuration is not available"错误
- 修复KeyError 'train_loss'：checkpoint保存时使用正确的metrics键名'total_loss'

### Added
- 完整的损失函数模块 (`loss.py`)：基于TiTok Stage 2实现
- 感知损失 (Perceptual Loss)：使用VGG网络特征
- 简化和完整两种损失函数选项
- 可选GAN对抗训练支持
- Einops库集成用于优雅的张量操作
- 扩展的CSV日志：记录所有损失组件
- H800 GPU集群的SBATCH作业脚本 (`titok_finetune_h800.sbatch`)
  - 配置1个H800 GPU，240GB内存，12 CPU核心
  - 使用q-hgpu-batch分区（批量作业专用）
  - 指定gpucluster-g4节点（当前可用H800节点）
  - 设置7天时间限制和邮件通知
  - 包含完整的环境检查和错误处理
  - 使用完整的TiTok损失函数（重建+感知+GAN）
  - 优化批大小为16，适合单GPU训练

### Changed
- 训练脚本集成新损失函数：替换简单的MSE损失
- 增强metrics记录：支持多组件损失跟踪
- 改进命令行参数：添加损失函数配置选项

## [1.2.6] - 2025-10-18

### Changed
- 修改demo.py默认输出目录：从`output/`更改为`outputs/demos/`以更好地组织演示结果
- 更新输出路径提示：显示完整的绝对路径以便用户轻松定位生成的文件

## [1.2.6] - 2025-10-18

### Fixed
- 修复CSV metrics日志记录：正确处理训练和验证的不同metrics键名
- 修复验证函数：现在返回完整的metrics字典包括compression_ratio、SSIM等
- 解决metrics为空的问题：确保所有计算的metrics都被正确记录

### Enhanced
- 改进validate函数：累积所有可用的metrics (MSE, PSNR, SSIM, LPIPS等)
- 优化CSV写入逻辑：正确映射训练和验证阶段的不同metrics名称
- 增强进度条显示：显示所有可用metrics的实时更新

## [1.2.1] - 2025-10-18

### Changed
- 修改demo.py脚本支持使用本地tokenizer权重：添加对tokenizer_titok_bl128_vae_c16_imagenet本地模型的自动检测和加载
- 优化模型加载逻辑：优先使用本地checkpoint，如果不存在则回退到HuggingFace预训练模型
- 提升模型灵活性：允许用户轻松切换使用自定义训练的tokenizer而无需修改代码

### Enhanced
- 改进模型加载的用户体验：添加明确的加载状态提示，显示使用本地还是远程模型
- 增强代码健壮性：添加路径存在性检查，避免加载失败时的崩溃

## [1.2.0] - 2025-10-18

### Fixed
- 修复训练时的梯度计算错误："element 0 of tensors does not require grad"
- 确保tokenizer参数在训练时可计算梯度
- 添加详细的调试信息和错误检查
- 强制模型处于训练模式以启用梯度计算

### Enhanced
- 添加参数统计日志：显示总参数数和可训练参数数
- 添加loss梯度检查：在反向传播前验证loss tensor
- 改进错误消息：提供具体的故障排除指导

### Debug
- 添加训练模式强制设置：model.train() 和 tokenizer.train()
- 增加参数可训练性验证

## [1.0.9] - 2025-10-18

### Added
- 专注CUDA demo：默认设备改为cuda，添加CUDA优先逻辑和警告
- 添加图像保存功能：保存验证和测试的输入/重建图像样本到输出目录
- 新增命令行参数：--save_images 和 --save_image_every 用于控制图像保存
- 测试集最终评估：训练完成后自动在测试集上评估并保存图像样本

### Changed
- 更新README.md突出CUDA训练命令，添加图像保存示例
- 优化设备选择逻辑：自动检测CUDA可用性并提供相应提示

## [1.0.8] - 2025-10-18

### Changed
- 大幅简化med_mri/README.md，专注于脚本启动命令指导
- 移除详细的API文档、高级用法示例和引用资料
- 保留核心命令示例和基本配置说明
- 精简故障排除部分，只保留最常见的三个问题
- 更新finetune_titok_mri.py默认数据集路径为相对路径 '../acdc_img_datasets'

## [1.0.7] - 2025-10-18

### Changed
- 更新med_mri/README.md中的脚本命令路径，适配从med_mri目录直接调用脚本的需求
- 修正数据集路径引用，从绝对路径改为相对路径（../acdc_img_datasets/）
- 更新安装和测试命令，使用正确的项目路径（/userhome/cs3/fung0311/CVPR-2025/MedCompression）
- 简化脚本调用方式，移除不必要的目录导航命令

## [1.0.6] - 2025-10-19

### Fixed
- 修复 git submodule 1d-tokenizer 引用错误：原引用提交 eae95054f9625a1f4e165aed45e15e55fa6a5f2d 在远程仓库中不存在，已更新为有效的最新提交 942a96f
- 解决 `git submodule update --init --recursive` 命令执行失败的问题
- 修复 ACDC 数据集转换器：脚本现在能正确识别 training/ 和 testing/ 子目录中的患者文件夹，支持标准 ACDC 数据集结构
- 改进 ACDC 转换器：自动排除 ground truth 分割掩码文件（*_gt.nii.gz），只转换原始 MRI 图像用于压缩训练

### Added
- ACDC 数据集转换器现在支持完整的 150 个患者（100 个训练 + 50 个测试），成功转换 2978 张纯 MRI 图像（排除分割掩码）
- 简化的 GPU 分配系统：添加 SLURM 到 PATH，默认使用 `srun --gres=gpu:1 --mail-type=ALL --pty bash`

## [1.0.0] - 2025-10-16

### Added
- 新增TiTok模型在医疗MRI数据压缩中的应用研究
- 添加ACDC心脏MRI数据集分析和处理方案
- 设计多模态融合策略用于MRI压缩
- 创建中文技术文档 README_MRI_TiTok.md
- 创建完整的MRI迁移分析文档 titok-mri.md，包含4个核心问题分析和mermaid可视化

### Changed
- 将TiTok模型从ImageNet迁移到医疗MRI数据压缩
- 适配4D MRI数据的时空特性
- 优化模型架构以支持单通道医疗图像

### Fixed
- 修正MRI数据集元数据可用性描述，明确指出临床参数不是所有数据集都包含

### Refactor
- 合并文档文件：将 mri_titok_analysis.md 合并到 titok-mri.md 中，统一文档结构
- 优化mermaid图表布局：改为垂直布局，提高可读性，添加多行文本和视觉元素

### Added
- 新增第5章"分割引导压缩"：设计轻量级分割网络，实现分割输出注入TiTok模型
- 添加ACDC分割数据格式分析和4类心脏结构标注说明
- 设计分割网络架构（基于ResNet18）和两种注入策略（注意力引导+联合训练损失）
- 提供完整的两阶段训练流程和性能提升预期

### Details
- **数据集分析**: 深入分析ACDC数据集的4D结构和临床信息
- **模态设计**: 识别空间、时间、分割和元数据四种关键模态
- **迁移策略**: 设计渐进式微调和模态适配器
- **优势比较**: 与传统压缩方法的定量对比分析
- **可视化**: 使用mermaid图表详细展示数据集结构、模态关系和方法对比

### 1.0.0

*   **Added**: 解释了 `train_titok.py` 中 TiTok 的训练方案，包括两个主要阶段：VQD-VAE 训练和生成模型训练。详细说明了每个阶段的假设/目的、环境配置、输入、输出和损失函数，并评估了其影响。

### 1.0.1

*   **Changed**: 详细阐明了 `train_titok.py` 和 `titok.py` 中 TiTok 模型在两个训练阶段（VQD-VAE 训练和生成模型训练）中哪些具体模块（包括编码器、量化器、解码器、潜在 token、像素量化器/解码器和判别器）是训练的或冻结的。进一步明确了 `finetune_decoder` 标志对模块训练状态的影响。

### 1.0.2

*   **Changed**: 详细阐述了 TiTok 模型在两个训练阶段中各自使用的损失函数。阶段一（VQD-VAE 训练）的损失包括重建损失、量化损失、对抗性损失、感知损失和潜在空间损失。阶段二（生成模型训练）的损失主要为交叉熵损失，并可能辅以对抗性损失、重建损失、量化损失和感知损失。

### 1.0.3

*   **Added**: 创建了新的 `README.md` 文件，详细介绍了 TiTok-MRI 项目的概览、核心特性（包括 TiTok 图像编解码、关键帧分割信息的中层融合、DiT 解码器结构）、预训练/微调/训练计划、环境配置与使用、预期性能指标、贡献与反馈以及许可证信息。
*   **影响评估**: `README.md` 文件的创建提供了项目的高级概述和使用指南，极大提升了项目的可理解性和易用性，便于新用户快速上手和理解项目。

### 1.0.4

*   **Changed**: 更新了 `README.md` 文件，重点阐述了从原始 TiTok 架构平滑过渡到 DiT 解码器的策略，并强调了将预训练 TiTok 模型用作高效图像 tokenizer 的核心作用。详细描述了 TiTok 编解码、关键帧分割信息的中层融合、DiT 解码器结构，以及调整后的多阶段训练计划，以更好地集成这些组件。
*   **影响评估**: 此次 `README.md` 的更新显著提高了项目描述的清晰度和连贯性，尤其在解释技术演进和模块集成方面。这将帮助用户更好地理解项目的设计哲学和实现细节。


