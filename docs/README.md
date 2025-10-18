# 🏥 TiTok-MRI：基于TiTok与DiT的医疗MRI数据高效压缩与重建

## 🚀 项目概览

本项目旨在 leveraging (利用) 先进的1D图像标记化（1D Image Tokenization, TiTok）模型和Diffusion Transformer（DiT）解码器，实现医疗MRI数据（特别是心脏4D MRI）的极高压缩比和高保真重建。通过创新性地将**预训练TiTok作为强大的离散图像tokenizer**，并结合**DiT解码器**进行像素级重建，我们致力于用少量离散token高效表示完整的4D心脏MRI序列，同时深度整合临床分割信息，实现从现有架构到前沿扩散模型的平滑过渡。

## ✨ 核心特性

### 1. 🖼️ 预训练TiTok作为高效的图像Tokenizer

我们充分利用**预训练的TiTok模型**作为核心的图像tokenizer。这包括其强大的**编码器（Encoder）**和**向量量化器（Vector Quantizer）**。
*   **编码器（Encoder）**: 负责将原始的连续MRI图像（每个时间帧）转换为紧凑的连续潜在表示。
*   **向量量化器（Vector Quantizer）**: 将编码器输出的连续潜在表示量化为固定大小的离散token序列。这些离散token是图像高度压缩、语义丰富的表示，为后续的生成任务提供了统一的离散接口。

**平滑过渡**: TiTok的编码器和量化器经过大规模自然图像数据集（如ImageNet）的预训练，具备强大的通用特征提取能力。在本项目中，它们作为**冻结的预训练Tokenizer**，直接提供高质量的离散token输入，实现了从原始TiTok架构到DiT解码器的无缝衔接，极大地加速了医疗领域的模型适应。

### 2. 💖 关键帧分割信息的中层融合

为了确保压缩和重建过程对心脏关键解剖结构的高度保真，我们整合了ED（收缩末期）和ES（舒张末期）两个关键时相的分割信息。我们采用**中层融合（Mid-level Fusion）**策略，在token表示层面进行融合：

*   **分割网络**: 使用轻量级分割网络（如基于ResNet18）对MRI关键帧进行精确的心脏结构（如左心室腔、心肌、右心室腔）分割
SAM3 - medical data - segemtation  (vision features + img = early/mid fushion)

*   **特征提取与条件化**: 从分割结果中提取形态学或概率特征，并将其作为**条件嵌入**，通过**交叉注意力机制**注入到DiT解码器的**Transformer层**中。这使得DiT在从token重建图像时，能够“感知”并优先关注关键解剖区域，确保结构完整性。
*   **联合训练损失**: 在训练过程中，除了DiT的扩散损失，还结合了分割一致性损失和临床相关损失（如心脏体积保持损失），确保模型在重建高质量图像的同时，优化了分割准确性和重要的临床指标。

这种融合方式确保了在高度压缩的情况下，对临床诊断至关重要的解剖结构信息能够被精确地保留和重建。

### 3. 🧠 DiT (Diffusion Transformer) 解码器结构

img per frame
200k
DiT 32x1x1024xT -> contrastive 

为了实现从离散token到高质量MRI图像的极致重建，我们将传统的TiTok解码器升级为**DiT（Diffusion Transformer）结构的解码器**。

*   **Diffusion Model的强大生成能力**: DiT解码器利用了扩散模型强大的图像生成潜力。它通过迭代去噪过程，逐步从一个随机噪声图像中恢复出高质量的重建结果。
*   **Transformer的序列建模优势**: DiT核心的Transformer结构使其能够更好地理解和利用由预训练TiTok tokenizer提供的离散token序列，通过自注意力机制捕捉token间的长距离依赖和全局上下文信息，从而指导高效的去噪和重建。
*   **平滑过渡**: DiT解码器直接接收来自**冻结TiTok tokenizer**的离散token，并结合条件信息（如分割特征）进行图像生成。这实现了从离散token表示到连续图像像素的平滑、高质量转换，避免了传统解码器在处理高压缩比时可能遇到的细节丢失和伪影问题。
*   **优势**: 这种架构能够提供卓越的图像重建质量，尤其是在处理高压缩比和复杂医疗图像细节时，DiT展现出更强的稳定性和精细度，有助于生成更符合临床标准的MRI图像。

### 4. 📚 预训练、微调与训练计划

我们的训练策略采用多阶段方法，旨在充分利用预训练模型的优势并逐步适应医疗MRI数据特性，最终实现DiT解码器与TiTok tokenizer的协同工作。

#### 可用的预训练模型
*   **ImageNet预训练的TiTok Tokenizer**: 这是我们方法的基础。利用在大型自然图像数据集（如ImageNet）上预训练的TiTok模型的**编码器和量化器**作为我们的**冻结tokenizer**。它提供了强大的通用图像特征提取和离散化能力。
*   **HuggingFace Hub上的TiTok权重**: 参考 `1d-tokenizer/scripts/train_titok.py` 中的 `hf_hub_download`，可以从HuggingFace下载预训练的TiTok模型权重，用于初始化我们的tokenizer部分。

#### 多阶段训练计划
我们采用三阶段训练策略以实现最佳性能，并确保从预训练TiTok到DiT解码器的平滑集成：

*   **阶段一：预训练TiTok Tokenizer的适应与冻结 (Stage 1: TiTok Tokenizer Adaptation & Freezing)**
    *   **目的**: 将预训练的TiTok编码器和量化器初步适应医疗MRI数据的基本特征。此阶段结束后，Tokenizer将被冻结，作为后续DiT训练的稳定输入源。
    *   **训练模块**: **主要训练** MRI数据特性的**通道适配器**和**时间建模模块**，以及对**TiTok编码器/量化器进行少量微调**（如果需要）。此阶段的损失主要关注**重建损失**和**token质量损失**。
    *   **状态**: 完成后，`TiTokEncoder` 和 `VectorQuantizer` 被**冻结**。

*   **阶段二：DiT解码器与分割融合训练 (Stage 2: DiT Decoder & Segmentation Fusion Training)**
    *   **目的**: 核心阶段，专注于训练DiT解码器以及分割信息融合模块，实现从离散token到高质量MRI图像的条件生成。
    *   **训练模块**: **冻结** TiTok tokenizer。**训练** **DiT解码器**，**交叉注意力融合层**和**轻量级分割网络**。DiT解码器接收来自冻结tokenizer的token作为输入。
    *   **损失**: 采用**层次化损失融合**，包括DiT的扩散损失、像素级重建损失（MSE/L1）、结构级分割损失（Dice/IoU）、功能级周期一致性损失，以及临床级疾病特异性损失。

*   **阶段三：端到端联合优化 (Stage 3: End-to-End Joint Optimization - 可选)**
    *   **目的**: 对整个 TiTok-MRI 模型（冻结Tokenizer + DiT解码器 + 融合模块）进行全局微调，以最大化其整体压缩效率和重建质量。
    *   **训练模块**: **选择性解冻** 部分关键模块（如 DiT 解码器的特定层或融合层），进行**联合训练**。
    *   **损失**: 综合所有损失函数，进行端到端优化，以达到最终的性能目标。这一阶段旨在进一步提升模型的泛化能力和鲁棒性。

## ⚙️ 环境配置与使用

```bash
# 安装核心依赖
pip install -r requirements.txt
# 额外医疗图像处理依赖
pip install nibabel scikit-image medpy accelerate transformers omegaconf einops torch torchvision
```

### 数据预处理
将ACDC MRI数据集转换为模型可识别的格式（例如WebDataset），并进行必要的归一化和数据增强。
```bash
python scripts/convert_acdc_to_wds.py \
    --input_dir /path/to/acdc_dataset \
    --output_dir acdc_wds \
    --modality "4d_mri"
```

### 模型训练
使用 `accelerate` 工具启动多GPU训练。请根据您的具体配置调整YAML文件路径和输出目录。
```bash
# 阶段一: TiTok Tokenizer 适应性训练 (示例配置路径)
# 此阶段结束后，TiTok编码器和量化器将被冻结
accelerate launch scripts/train_mri_titok.py \
    config=configs/training/MRI/stage1/mri_titok_tokenizer_adapt.yaml \
    experiment.output_dir="mri_titok_tokenizer_stage1_output"

# 阶段二: DiT解码器与模态融合训练 (加载阶段一权重)
accelerate launch scripts/train_mri_titok.py \
    config=configs/training/MRI/stage2/mri_titok_dit_fusion.yaml \
    experiment.init_weight="path/to/mri_titok_tokenizer_stage1_output/checkpoint_last.pth" \
    experiment.output_dir="mri_titok_dit_fusion_stage2_output"

# 阶段三: 端到端联合优化 (可选，加载阶段二权重)
accelerate launch scripts/train_mri_titok.py \
    config=configs/training/MRI/stage3/mri_titok_e2e.yaml \
    experiment.init_weight="path/to/mri_titok_dit_fusion_stage2_output/checkpoint_last.pth" \
    experiment.output_dir="mri_titok_e2e_stage3_output"
```

### 推理使用
加载训练好的模型进行MRI的压缩与重建。
```python
from modeling.titok import TiTok
# 加载完整模型，确保TiTok tokenizer和DiT解码器的配置正确
mri_model = TiTok.from_pretrained("path/to/mri_titok_e2e_stage3_output")
mri_model.eval() # 切换到评估模式

# 压缩4D MRI序列
# 假设 mri_4d_tensor 是预处理好的4D MRI数据 (Batch, Channel, H, W, D, T)
# TiTok encode 方法将处理时间维度或通过循环处理单帧
with torch.no_grad():
    compressed_tokens = mri_model.encode(mri_4d_tensor)  # 返回离散token，如 (Batch, num_tokens)

# 重建完整4D MRI
with torch.no_grad():
    reconstructed_mri = mri_model.decode_tokens(compressed_tokens) # 返回重建的4D MRI tensor
```

## 📈 预期性能指标 (示例)

*   **重建保真度**: PSNR > 38dB, SSIM > 0.96
*   **临床准确性**: 心脏关键结构分割Dice > 0.92
*   **压缩效率**: 100-1000x 压缩比，单个4D MRI序列可压缩至32个token
*   **推理速度**: < 5ms/帧 (解码)

## 🤝 贡献与反馈

欢迎任何形式的贡献、建议和反馈。如果您在使用过程中遇到问题或有改进想法，请提交 issue 或 pull request。

## 📜 许可证

本项目基于 Apache-2.0 许可证开源。

**注意**: 本项目仅用于研究目的，请遵守医疗数据隐私法规和相关伦理准则。
