# 🔧 Git Submodule Management Guide

## 📋 Overview

This project uses git submodules to manage external dependencies, specifically the `1d-tokenizer` repository from ByteDance.

## 🏗️ Current Submodules

### 1d-tokenizer
- **Repository**: https://github.com/bytedance/1d-tokenizer.git
- **Path**: `1d-tokenizer/`
- **Purpose**: TiTok tokenization and compression framework
- **Custom additions**: MRI integration and medical imaging tools

## 🚀 Working with Submodules

### Initial Setup
```bash
# Clone the main repository with submodules
git clone --recurse-submodules https://github.com/Justinfungi/ICLR-Med-Compression.git

# Or if already cloned, initialize submodules
git submodule update --init --recursive
```

### Updating Submodules
```bash
# Update to latest upstream version
cd 1d-tokenizer
git pull origin main

# Return to main repo and commit the update
cd ..
git add 1d-tokenizer
git commit -m "Update 1d-tokenizer submodule"
```

### Making Changes to Submodules
```bash
# Navigate to submodule
cd 1d-tokenizer

# Make your changes
# ... edit files ...

# Commit changes in submodule
git add .
git commit -m "Your changes"

# Return to main repo and update reference
cd ..
git add 1d-tokenizer
git commit -m "Update submodule with custom changes"
```

## 📁 Submodule Structure

### What's Included
The `1d-tokenizer` submodule includes:
- ✅ All source code (`.py` files)
- ✅ Configuration files (`.yaml`, `.json`)
- ✅ Documentation (`.md` files)
- ✅ Scripts and utilities
- ✅ Custom MRI integration (`med_mri/`)
- ✅ Demo files (`demo_mri.py`)

### What's Excluded (.gitignore)
The submodule excludes large files:
- ❌ Model checkpoints (`checkpoints/`)
- ❌ Training data (`data/`)
- ❌ Cache files (`.cache/`)
- ❌ Binary model files (`.bin`, `.safetensors`)
- ❌ Test outputs (`test_finetune*/`)

## 🔄 Workflow Examples

### Adding New Features to Submodule
```bash
# 1. Navigate to submodule
cd 1d-tokenizer

# 2. Create feature branch (optional)
git checkout -b feature/new-mri-tool

# 3. Make changes
echo "# New MRI Tool" > med_mri/new_tool.py

# 4. Commit in submodule
git add .
git commit -m "Add new MRI processing tool"

# 5. Return to main repo
cd ..

# 6. Update submodule reference
git add 1d-tokenizer
git commit -m "Add new MRI tool to 1d-tokenizer submodule"
```

### Syncing with Upstream
```bash
# 1. Navigate to submodule
cd 1d-tokenizer

# 2. Add upstream remote (if not already added)
git remote add upstream https://github.com/bytedance/1d-tokenizer.git

# 3. Fetch upstream changes
git fetch upstream

# 4. Merge or rebase with upstream
git merge upstream/main
# or
git rebase upstream/main

# 5. Resolve any conflicts with custom changes

# 6. Return to main repo and update
cd ..
git add 1d-tokenizer
git commit -m "Sync 1d-tokenizer with upstream"
```

## 🛠️ Troubleshooting

### Submodule Not Initialized
```bash
git submodule update --init --recursive
```

### Detached HEAD in Submodule
```bash
cd 1d-tokenizer
git checkout main  # or your preferred branch
cd ..
git add 1d-tokenizer
git commit -m "Fix submodule HEAD"
```

### Merge Conflicts in Submodule
```bash
cd 1d-tokenizer
# Resolve conflicts manually
git add .
git commit -m "Resolve merge conflicts"
cd ..
git add 1d-tokenizer
git commit -m "Update submodule after conflict resolution"
```

### Reset Submodule to Clean State
```bash
# Reset submodule to committed version
git submodule update --force

# Or reset to specific commit
cd 1d-tokenizer
git reset --hard <commit-hash>
cd ..
git add 1d-tokenizer
git commit -m "Reset submodule to specific version"
```

## 📊 Best Practices

### ✅ Do's
- Always commit submodule changes before updating main repo
- Use descriptive commit messages for submodule updates
- Keep custom changes in separate directories (`med_mri/`)
- Regularly sync with upstream to get latest features
- Test thoroughly after submodule updates

### ❌ Don'ts
- Don't commit large files to submodules
- Don't modify core upstream files directly
- Don't forget to update main repo after submodule changes
- Don't push submodule changes to upstream without permission

## 🔍 Monitoring Submodule Status

### Check Submodule Status
```bash
# Show submodule status
git submodule status

# Show detailed submodule info
git submodule foreach git status

# Check for uncommitted changes
git status --submodule
```

### Useful Git Aliases
```bash
# Add to ~/.gitconfig
[alias]
    sub-status = submodule foreach git status
    sub-pull = submodule foreach git pull
    sub-update = submodule update --remote --merge
```

## 📚 Additional Resources

- [Git Submodules Documentation](https://git-scm.com/book/en/v2/Git-Tools-Submodules)
- [1d-tokenizer Repository](https://github.com/bytedance/1d-tokenizer)
- [TiTok Paper](https://arxiv.org/abs/2310.05737)

## 🤝 Contributing

When contributing to submodules:
1. Fork the upstream repository if making significant changes
2. Keep custom medical imaging code in `med_mri/` directory
3. Document all changes in commit messages
4. Test compatibility with main project
5. Consider contributing useful features back to upstream
