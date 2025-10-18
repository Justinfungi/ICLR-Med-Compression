#!/bin/bash

# Simple GPU Runner for TiTok MRI
# Usage: ./gpu_run.sh [command]
# Run from med_mri/ directory: cd med_mri && ./gpu_run.sh test

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add SLURM to PATH
export PATH="/usr/local/slurm/bin:$PATH"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🚀 TiTok MRI GPU Runner (from $SCRIPT_DIR)${NC}"

case ${1:-help} in
    test)
        echo -e "${GREEN}Running test on GPU...${NC}"
        srun --gres=gpu:1 --mail-type=ALL --pty bash -c "
            cd '$PROJECT_ROOT'
            conda activate cvpr2025-py39
            python med_mri/run_test.py
        "
        ;;
    train)
        echo -e "${GREEN}Starting training on GPU...${NC}"
        srun --gres=gpu:1 --mail-type=ALL --pty bash -c "
            cd '$PROJECT_ROOT'
            conda activate cvpr2025-py39
            python med_mri/finetune_titok_mri.py
        "
        ;;
    bash)
        echo -e "${GREEN}Starting GPU bash session...${NC}"
        srun --gres=gpu:1 --mail-type=ALL --pty bash -c "
            cd '$PROJECT_ROOT'
            echo 'Welcome to GPU environment!'
            echo 'Project root: $PWD'
            exec bash
        "
        ;;
    help|*)
        echo "Usage: $0 {test|train|bash}"
        echo "  test  - Run test on GPU"
        echo "  train - Start training on GPU"
        echo "  bash  - Start interactive GPU bash in project root"
        echo ""
        echo "Run from med_mri/ directory:"
        echo "  cd med_mri && ./gpu_run.sh test"
        ;;
esac
