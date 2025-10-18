#!/usr/bin/env python3
"""
ACDC数据集帧导出演示脚本

展示如何使用export_all_frames.py导出不同类型的图像数据
"""

import os
import sys
from pathlib import Path

def run_export_demo():
    """运行导出演示"""

    print("🖼️ ACDC数据集帧导出演示")
    print("=" * 50)

    # 检查脚本是否存在
    export_script = Path(__file__).parent / "export_all_frames.py"
    if not export_script.exists():
        print(f"❌ 导出脚本不存在: {export_script}")
        return False

    demos = [
        {
            "name": "4D时序数据导出演示",
            "description": "导出前2个患者的完整心跳周期4D数据",
            "command": f"python export_all_frames.py --mode 4d_sequence --end_idx 2 --verbose"
        },
        {
            "name": "关键帧导出演示",
            "description": "导出前1个患者的ED和ES关键帧所有切片",
            "command": f"python export_all_frames.py --mode 3d_keyframes --end_idx 1 --verbose"
        },
        {
            "name": "ED时相导出演示",
            "description": "只导出前1个患者的ED时相所有切片",
            "command": f"python export_all_frames.py --mode ed_only --end_idx 1 --verbose"
        }
    ]

    for i, demo in enumerate(demos, 1):
        print(f"\n🎯 演示 {i}: {demo['name']}")
        print(f"📝 {demo['description']}")
        print(f"💻 命令: {demo['command']}")
        print("-" * 50)

        # 询问用户是否运行
        response = input("是否运行此演示? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            print("🚀 执行演示...")
            os.system(demo['command'])
            print(f"✅ 演示 {i} 完成\n")
        else:
            print(f"⏭️ 跳过演示 {i}\n")

    print("🎉 所有演示完成!")
    print("\n📁 查看导出的图像:")
    print("   ls -la acdc_img_datasets/")
    print("\n📖 阅读详细文档:")
    print("   cat EXPORT_README.md")

    return True

def show_usage_examples():
    """显示使用示例"""

    print("\n💡 常用使用示例:")
    print("-" * 30)

    examples = [
        "# 导出所有患者的4D时序数据",
        "python export_all_frames.py --mode 4d_sequence",
        "",
        "# 导出前10个患者的关键帧",
        "python export_all_frames.py --mode 3d_keyframes --end_idx 10",
        "",
        "# 导出到自定义目录",
        "python export_all_frames.py --target_dir ../my_images --mode 4d_sequence",
        "",
        "# 不创建子目录（所有图像在一个目录）",
        "python export_all_frames.py --no_subdirs --mode ed_only",
        "",
        "# 详细输出模式",
        "python export_all_frames.py --verbose --end_idx 3"
    ]

    for example in examples:
        print(example)

def main():
    """主函数"""
    try:
        success = run_export_demo()
        if success:
            show_usage_examples()
        return 0
    except KeyboardInterrupt:
        print("\n\n👋 用户中断演示")
        return 0
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())



