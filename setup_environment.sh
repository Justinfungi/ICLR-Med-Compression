#!/bin/bash

# ===========================================
# Conda环境管理脚本
# 用于修复和验证conda环境配置
# ===========================================

echo "🔧 Conda环境修复和验证工具"
echo "================================="

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 函数：显示状态
show_status() {
    echo -e "${YELLOW}当前状态:${NC}"
    echo "  CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
    echo "  Python版本: $(python --version 2>&1)"
    echo "  Python路径: $(which python)"
    echo "  PATH包含conda: $(echo $PATH | grep -o '/anaconda3' | wc -l) 条目"
}

# 函数：重置环境
reset_environment() {
    echo -e "${YELLOW}重置环境变量...${NC}"

    # 退出所有conda环境
    conda deactivate 2>/dev/null || true
    conda deactivate 2>/dev/null || true

    # 清除conda相关环境变量
    unset CONDA_DEFAULT_ENV
    unset CONDA_PREFIX
    unset CONDA_PYTHON_EXE
    unset CONDA_EXE

    # 重置PATH（移除所有conda路径）
    export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/userhome/cs3/fung0311/.local/bin"

    # 重新添加base conda到PATH
    export PATH="/userhome/cs3/fung0311/anaconda3/bin:/userhome/cs3/fung0311/anaconda3/condabin:$PATH"

    echo -e "${GREEN}环境已重置${NC}"
}

# 函数：激活环境
activate_environment() {
    local env_name=$1
    if [ -z "$env_name" ]; then
        env_name="base"
    fi

    echo -e "${YELLOW}激活环境: $env_name${NC}"

    if [ "$env_name" = "base" ]; then
        # 激活base环境
        export PATH="/userhome/cs3/fung0311/anaconda3/bin:/userhome/cs3/fung0311/anaconda3/condabin:$PATH"
        export CONDA_DEFAULT_ENV="base"
        export CONDA_PREFIX="/userhome/cs3/fung0311/anaconda3"
    else
        # 激活其他环境
        local env_path="/userhome/cs3/fung0311/anaconda3/envs/$env_name"
        if [ ! -d "$env_path" ]; then
            echo -e "${RED}环境不存在: $env_path${NC}"
            return 1
        fi

        export PATH="$env_path/bin:/userhome/cs3/fung0311/anaconda3/bin:/userhome/cs3/fung0311/anaconda3/condabin:$PATH"
        export CONDA_DEFAULT_ENV="$env_name"
        export CONDA_PREFIX="$env_path"
    fi

    echo -e "${GREEN}成功激活环境: $env_name${NC}"
}

# 函数：验证环境
verify_environment() {
    local expected_env=$1

    echo -e "${YELLOW}验证环境配置...${NC}"

    # 检查CONDA_DEFAULT_ENV
    if [ "$CONDA_DEFAULT_ENV" != "$expected_env" ]; then
        echo -e "${RED}❌ CONDA_DEFAULT_ENV不匹配: 期望 $expected_env, 实际 $CONDA_DEFAULT_ENV${NC}"
        return 1
    fi

    # 检查python路径是否在正确的环境目录中
    local python_path=$(which python)
    if [[ $python_path != *"/anaconda3/envs/$expected_env/bin/python" ]] && [[ $expected_env == "base" && $python_path != *"/anaconda3/bin/python" ]]; then
        echo -e "${RED}❌ Python路径不正确: $python_path${NC}"
        return 1
    fi

    echo -e "${GREEN}✅ 环境配置正确${NC}"
    return 0
}

# 主逻辑
case "${1:-help}" in
    "status")
        show_status
        ;;
    "reset")
        reset_environment
        show_status
        ;;
    "activate")
        env_name="${2:-base}"
        reset_environment
        activate_environment $env_name
        show_status
        verify_environment $env_name
        ;;
    "verify")
        env_name="${2:-$CONDA_DEFAULT_ENV}"
        if [ -z "$env_name" ]; then
            env_name="base"
        fi
        verify_environment $env_name
        ;;
    "fix")
        echo "自动修复环境配置..."
        reset_environment
        activate_environment "base"
        if verify_environment "base"; then
            echo -e "${GREEN}环境修复成功！${NC}"
        else
            echo -e "${RED}环境修复失败，请手动检查${NC}"
        fi
        ;;
    "help"|*)
        echo "用法: $0 <command> [args]"
        echo ""
        echo "命令:"
        echo "  status                 显示当前环境状态"
        echo "  reset                  重置环境变量"
        echo "  activate <env>         激活指定环境 (默认: base)"
        echo "  verify <env>           验证环境配置 (默认: 当前环境)"
        echo "  fix                    自动修复环境配置"
        echo "  help                   显示此帮助信息"
        echo ""
        echo "示例:"
        echo "  $0 status"
        echo "  $0 activate TEXTure"
        echo "  $0 verify ICLR-2025-FedLearning"
        echo "  $0 fix"
        ;;
esac
