# 重新加载配置
source ~/.bashrc

# 直接使用 srun
srun --gres=gpu:1 --mail-type=ALL --pty bash

# 或使用别名 (如果设置了)
gpu