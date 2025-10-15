#!/bin/bash

# ===========================================
# 环境修复脚本 - 解决Python路径与conda环境不一致问题
# ===========================================

echo "🔧 修复Python路径与conda环境不一致问题..."
echo "================================="

# 函数：显示当前状态
show_status() {
    echo "当前状态:"
    echo "  CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
    echo "  Python版本: $(python --version 2>&1)"
    echo "  Python路径: $(which python)"
}

# 强制清理PATH并设置正确的环境
fix_environment() {
    echo "正在修复环境..."

    # 1. 清理conda相关环境变量
    unset CONDA_DEFAULT_ENV
    unset CONDA_PREFIX
    unset CONDA_EXE
    unset CONDA_PYTHON_EXE
    unset CONDA_SHLVL

    # 2. 完全清理PATH中的conda环境路径
    export PATH=$(echo "$PATH" | sed 's|/userhome/cs3/fung0311/anaconda3/envs/[^:]*:||g' | sed 's|::|:|g' | sed 's|^:||' | sed 's|:$||')

    # 3. 设置基础PATH（不包含任何conda环境）
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/userhome/cs3/fung0311/.local/bin"

    # 4. 添加base conda路径
    export PATH="/userhome/cs3/fung0311/anaconda3/bin:/userhome/cs3/fung0311/anaconda3/condabin:$PATH"

    # 5. 设置conda环境变量为base
    export CONDA_DEFAULT_ENV="base"
    export CONDA_PREFIX="/userhome/cs3/fung0311/anaconda3"

    echo "✅ 环境修复完成"
}

# 主逻辑
echo "修复前状态:"
show_status
echo

fix_environment
echo

echo "修复后状态:"
show_status
echo

# 验证修复结果
if [[ "$(which python)" == */anaconda3/bin/python ]] && [[ "$CONDA_DEFAULT_ENV" == "base" ]]; then
    echo "🎉 成功！Python路径现在与conda环境完美对齐！"
    echo ""
    echo "提示："
    echo "  - 要激活其他环境，请运行: conda activate <环境名>"
    echo "  - 要返回base环境，请运行: conda activate base"
    echo "  - 要自动修复环境，请运行: source fix_environment.sh"
else
    echo "❌ 修复失败，请联系管理员"
    exit 1
fi
