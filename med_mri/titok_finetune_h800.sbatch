#!/bin/bash

# TiTok MRI Fine-tuning Job for H800 GPUs
# 使用 q-hgpu-batch 分区 (H100/H800 80GB GPUs，批量作业专用，指定gpucluster-g4节点)

#SBATCH --job-name=titok_mri_h800
#SBATCH --output=titok_mri_%j.out
#SBATCH --error=titok_mri_%j.err

# 资源配置 - 使用1个H800 GPU (q-hgpu-batch分区，批量作业专用)
#SBATCH --partition=q-hgpu-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:h800:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=240GB
#SBATCH --nodelist=gpucluster-g4

# 时间限制 - 7天 (q-hgpu-batch最大限制)
#SBATCH -t 7-00:00:00

# 邮件通知
#SBATCH --mail-type=ALL
#SBATCH --mail-user=fung0311@connect.hku.hk

# 作业信息输出
echo "开始时间: $(date)"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURM_JOB_NODELIST"
echo "GPU数量: $SLURM_GPUS"
echo "分配CPU数: $SLURM_CPUS_PER_TASK"
echo "分配内存: $SLURM_MEM_PER_NODE"

# 初始化Anaconda
echo "初始化Anaconda环境..."
. $HOME/anaconda3/etc/profile.d/conda.sh

# 激活环境
echo "激活cvpr2025-py39环境..."
conda activate cvpr2025-py39

# 检查环境激活
if [ $? -ne 0 ]; then
    echo "错误: 无法激活conda环境 cvpr2025-py39"
    exit 1
fi

# 检查CUDA可用性
echo "检查CUDA和GPU状态..."
nvidia-smi
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"

if [ $? -ne 0 ]; then
    echo "错误: PyTorch或CUDA检查失败"
    exit 1
fi

# 进入工作目录
echo "进入工作目录..."
cd /userhome/cs3/fung0311/CVPR-2025/MedCompression/med_mri

if [ $? -ne 0 ]; then
    echo "错误: 无法进入工作目录"
    exit 1
fi

# 创建输出目录（如果不存在）
mkdir -p ./outputs/checkpoints

# 运行TiTok MRI微调
echo "开始TiTok MRI微调训练..."
echo "命令: python finetune_titok_mri.py --data_root ../acdc_img_datasets --output_dir ./outputs --tokenizer_path ./checkpoints/tokenizer_titok_bl128_vae_c16_imagenet --batch_size 16 --num_epochs 300 --learning_rate 1e-4 --save_every 10 --save_images --save_image_every 5 --device cuda --use_full_loss --use_gan --reconstruction_weight 1.0 --perceptual_weight 0.15 --reconstruction_loss_type l2 --perceptual_net_type vgg16 --discriminator_weight 0.8 --discriminator_start 2000"

python finetune_titok_mri.py \
    --data_root ../acdc_img_datasets \
    --output_dir ./outputs \
    --tokenizer_path ./checkpoints/tokenizer_titok_bl128_vae_c16_imagenet \
    --batch_size 16 \
    --num_epochs 300 \
    --learning_rate 1e-4 \
    --save_every 10 \
    --save_images \
    --save_image_every 5 \
    --device cuda \
    --use_full_loss \
    --use_gan \
    --reconstruction_weight 1.0 \
    --perceptual_weight 0.15 \
    --reconstruction_loss_type l2 \
    --perceptual_net_type vgg16 \
    --discriminator_weight 0.8 \
    --discriminator_start 2000

# 检查退出状态
if [ $? -eq 0 ]; then
    echo "训练成功完成!"
else
    echo "训练失败，退出码: $?"
    exit 1
fi

echo "结束时间: $(date)"
echo "作业完成!"
