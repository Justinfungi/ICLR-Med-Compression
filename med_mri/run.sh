#!/bin/bash

# TiTok MRI Fine-tuning - Run Script
# Run from med_mri directory: ./run.sh test
#                             ./run.sh train --batch_size 4 --num_epochs 5

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print header
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}🫀 TiTok MRI Fine-tuning Module${NC}"
echo -e "${BLUE}======================================${NC}\n"

# Show available commands
show_help() {
    echo -e "${GREEN}Available commands:${NC}"
    echo "  test              - Run quick functionality test"
    echo "  train [args]      - Start training with optional arguments"
    echo "  help              - Show this help message"
    echo ""
    echo -e "${GREEN}Training arguments:${NC}"
    echo "  --batch_size N         Batch size (default: 8)"
    echo "  --num_epochs N         Number of epochs (default: 20)"
    echo "  --learning_rate LR     Learning rate (default: 1e-4)"
    echo "  --device DEVICE        Device: 'cpu' or 'cuda' (default: cuda)"
    echo "  --save_every N         Save checkpoint every N epochs (default: 5)"
    echo "  --output_dir DIR       Output directory (default: ./outputs)"
    echo ""
    echo -e "${GREEN}Examples:${NC}"
    echo "  ./run.sh test"
    echo "  ./run.sh train --batch_size 4 --num_epochs 5 --device cpu"
    echo "  ./run.sh train --batch_size 8 --num_epochs 20 --device cuda"
}

# Parse command
COMMAND=${1:-help}

case $COMMAND in
    test)
        echo -e "${GREEN}🧪 Running quick test...${NC}\n"
        python3 run_test.py
        echo -e "\n${GREEN}✅ Test completed!${NC}"
        ;;
    train)
        shift  # Remove 'train' from arguments
        echo -e "${GREEN}🏋️ Starting training...${NC}\n"
        python3 run_train.py "$@"
        echo -e "\n${GREEN}✅ Training completed!${NC}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $COMMAND${NC}\n"
        show_help
        exit 1
        ;;
esac
