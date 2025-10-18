#!/usr/bin/env python3
"""
Standalone test script - Run from med_mri directory
python run_test.py
"""

import sys
from pathlib import Path

# Add med_mri directory to path
med_mri_dir = Path(__file__).parent
sys.path.insert(0, str(med_mri_dir))
sys.path.insert(0, str(med_mri_dir.parent))

# Import and run test
from test_finetune import test_basic_functionality

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
