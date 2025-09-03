### ACDC 数据集下载指南（Hugging Face）

**目标**：从 `fan040701/ACDC` 同步数据到本地，用于医学压缩/分割等实验。

**环境准备**
- Python 3.8+
- 依赖：`huggingface_hub`、`tqdm`

```bash
pip install huggingface_hub tqdm
```

**脚本位置**：`MedCompression/scripts/acdc_download.py`

**快速开始**
```bash
# 请使用您本机的可写绝对路径（不要使用占位符 /abs/...）
python MedCompression/scripts/acdc_download.py \
  --output_dir /Users/<username>/Documents/ICLR/MedCompression/acdc_dataset
```

**访问令牌（私人/受限数据集）**
- 可通过命令行传入：`--hf_token YOUR_HF_TOKEN`
- 或使用环境变量：`HF_TOKEN` 或 `HF_HOME_TOKEN`

```bash
# 方式一：命令行参数
python MedCompression/scripts/acdc_download.py \
  --output_dir /Users/<username>/Documents/ICLR/MedCompression/acdc_dataset \
  --hf_token hf_xxx...

# 方式二：环境变量
export HF_TOKEN=hf_xxx...
python MedCompression/scripts/acdc_download.py \
  --output_dir /Users/<username>/Documents/ICLR/MedCompression/acdc_dataset
```

**常用参数**
- `--output_dir`: 本地展开目录（默认：当前工作目录的 `ACDC`）。
- `--cache_dir`: 指定 HF 缓存目录（可选）。
- `--allow_patterns`: 仅下载匹配的文件模式，可重复传入。
  - 例：`"**/*.nii.gz" "**/*.cfg" "**/*.md"`
- `--exclude_patterns`: 排除特定模式文件。
  - 例：`"**/.gitattributes"`
- `--only_metadata`: 仅同步元数据（提示：配合 `--allow_patterns` 精确控制）。
- `--revision`: 指定分支/标签/commit hash。

**示例：只下载影像与配置**
```bash
python MedCompression/scripts/acdc_download.py \
  --output_dir /Users/<username>/Documents/ICLR/MedCompression/acdc_dataset \
  --allow_patterns "**/*.nii.gz" "**/*.cfg" "**/*.md" \
  --exclude_patterns "**/.gitattributes"
```

**注意事项**
- 默认断点续传，重复执行不会重复下载已存在文件。
- 如果磁盘是 APFS/HFS，`local_dir_use_symlinks=False` 会进行复制/硬链接以增强兼容性。
- 数据集体量较大，请确保磁盘空间充足（>20GB 建议）。

**引用**
- 数据来源：`fan040701/ACDC`（Hugging Face）。


