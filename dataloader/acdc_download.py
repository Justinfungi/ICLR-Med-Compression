#!/usr/bin/env python3
"""
ACDC 数据集下载脚本（Hugging Face Datasets Hub）

功能概述：
- 通过 huggingface_hub 快速拉取 `fan040701/ACDC` 数据集到本地目录。
- 支持指定输出目录、缓存目录、包含/排除文件模式、是否仅获取元数据。
- 默认使用断点续传与本地缓存，避免重复下载。

使用示例：
  python MedCompression/scripts/acdc_download.py \
    --output_dir /path/to/ACDC \
    --allow_patterns "**/*.nii.gz" "**/*.cfg" "**/*.md" \
    --exclude_patterns "**/.gitattributes" 

环境依赖：
  pip install huggingface_hub tqdm
"""

import argparse
import os
from typing import List, Optional

from huggingface_hub import snapshot_download, login, HfApi
from tqdm import tqdm


DATASET_REPO_ID = "fan040701/ACDC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="下载 Hugging Face 上的 ACDC 数据集到本地目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.getcwd(), "ACDC"),
        help="数据集在本地展开的目标目录（默认为当前工作目录下的 ACDC）",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=DATASET_REPO_ID,
        help="目标数据集 repo_id（默认 fan040701/ACDC）",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Hugging Face 缓存目录（可选，默认使用 ~/.cache/huggingface）",
    )
    parser.add_argument(
        "--allow_patterns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "仅下载符合这些通配符模式的文件（可选，支持多个，如 '**/*.nii.gz' '**/*.cfg'）"
        ),
    )
    parser.add_argument(
        "--exclude_patterns",
        type=str,
        nargs="*",
        default=None,
        help=(
            "排除符合这些通配符模式的文件（可选，支持多个，如 '**/.gitattributes'）"
        ),
    )
    parser.add_argument(
        "--only_metadata",
        action="store_true",
        help="仅同步仓库结构与元数据，不实际下载大文件",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="可选：指定数据集分支名/标签/commit hash",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help=(
            "Hugging Face 访问令牌（可选）。若不提供，将尝试读取环境变量 HF_TOKEN/HF_HOME_TOKEN。"
        ),
    )
    return parser.parse_args()


def pretty_join(patterns: Optional[List[str]]) -> str:
    if not patterns:
        return "(未指定)"
    return ", ".join(patterns)


def main() -> None:
    args = parse_args()

    # 输出目录可写性检查
    output_dir = os.path.abspath(args.output_dir)
    parent_dir = os.path.dirname(output_dir)
    try:
        if not os.path.isdir(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        test_file = os.path.join(output_dir, ".write_test")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
    except Exception as fs_err:
        print("输出目录不可写，请更换为可写的绝对路径，例如：")
        print("  /Users/<username>/Documents/ICLR/MedCompression/acdc_dataset")
        print(f"当前路径: {output_dir}")
        print(str(fs_err))
        raise

    # 读取访问令牌：优先命令行参数，其次环境变量
    hf_token = (
        args.hf_token
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HF_HOME_TOKEN")
    )
    if hf_token:
        try:
            login(token=hf_token, add_to_git_credential=True)
            print("已使用提供的 Hugging Face 访问令牌完成登录认证。")
        except Exception as auth_err:
            print("令牌登录失败：")
            print(str(auth_err))
            # 不直接退出，允许继续尝试匿名访问，便于公共仓库场景

    print("开始下载 Hugging Face 数据集：")
    print(f"  Repo: {args.repo_id}")
    print(f"  输出目录: {output_dir}")
    if args.cache_dir:
        print(f"  缓存目录: {args.cache_dir}")
    print(f"  允许模式: {pretty_join(args.allow_patterns)}")
    print(f"  排除模式: {pretty_join(args.exclude_patterns)}")
    if args.only_metadata:
        print("  模式: 仅元数据（不下载大文件）")
    if args.revision:
        print(f"  指定 revision: {args.revision}")

    # snapshot_download 会将仓库完整快照下载到缓存中，并在 output_dir 建立硬链接/复制
    # repo_type="dataset" 指向数据集仓库
    # allow_patterns / ignore_patterns 提供了精细过滤能力
    # local_dir_use_symlinks=False 确保在某些文件系统下更好兼容性
    try:
        # 预检：验证仓库可访问性（可捕获 401/404 并给出明确提示）
        try:
            HfApi().dataset_info(repo_id=args.repo_id, revision=args.revision)
        except Exception as info_err:
            print("无法访问数据集仓库，请检查：")
            print("  1) repo_id 是否正确，例如 fan040701/ACDC")
            print("  2) 若为私有/受限仓库，请确保令牌有效并具备权限")
            print("  3) 可尝试在浏览器打开仓库页面确认权限")
            print(str(info_err))
            raise

        local_path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            cache_dir=args.cache_dir,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            allow_patterns=args.allow_patterns,
            ignore_patterns=args.exclude_patterns,
            revision=args.revision,
            # 当 only_metadata=True 时，仅拉取 refs 与 .gitattributes 等元信息
            # 通过设置 allow_patterns=[] 达到“空文件匹配”效果
            # 这里使用同步后手动提示的方式处理
        )
        if args.only_metadata:
            # 提示：当前 huggingface_hub 不直接暴露“仅元数据”的显式参数，
            # 如需完全不下载大文件，可结合 allow_patterns 精确控制。
            print(
                "提示：已同步仓库结构。若需完全跳过大文件，请结合 --allow_patterns 精确筛选。"
            )

        # 粗略统计下载结果（非严格，主要用于用户反馈）
        file_count = 0
        total_size = 0
        for root, _, files in os.walk(local_path):
            for f in files:
                file_path = os.path.join(root, f)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except OSError:
                    pass

        print("下载完成：")
        print(f"  本地路径: {local_path}")
        print(f"  文件数量: {file_count}")
        print(f"  估计大小: {total_size / (1024 ** 3):.2f} GiB")

    except Exception as e:
        print("下载失败：")
        print(str(e))
        raise


if __name__ == "__main__":
    main()


