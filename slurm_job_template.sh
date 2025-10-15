#!/bin/bash
#SBATCH --job-name=med_compression
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# ===========================================
# SLURM作业脚本 - 医学图像压缩项目
# ===========================================

# 设置环境变量（关键步骤）
export CONDA_AUTO_UPDATE_CONDA=false

# 初始化conda环境
eval "$(conda shell.bash hook)"

# 激活指定的conda环境
# 替换为您的环境名称，例如: TEXTure, ICLR-2025-FedLearning 等
CONDA_ENV="ICLR-2025-FedLearning"  # 修改为您需要的环境

echo "==========================================="
echo "激活conda环境: $CONDA_ENV"
echo "==========================================="

conda activate $CONDA_ENV

# 验证环境激活
echo "当前Python版本: $(python --version)"
echo "Python路径: $(which python)"
echo "Conda环境: $CONDA_DEFAULT_ENV"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==========================================="

# 设置工作目录
cd /userhome/cs3/fung0311/CVPR-2025/MedCompression
echo "工作目录: $(pwd)"

# 您的训练/推理命令
echo "开始执行任务..."

# 示例命令（根据您的需要修改）：
# python train.py --config config.yaml
# python scripts/acdc_download.py
# python dataloader/tests/example_usage.py

echo "==========================================="
echo "任务完成"
echo "==========================================="
