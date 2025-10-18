"""
批量导出所有ACDC患者帧为PNG图像

此脚本将加载ACDC数据集中的所有患者，为每个患者导出所有可用帧，
并使用标准命名约定保存为PNG格式。

功能特性:
- 支持ED/ES关键帧导出
- 支持4D时序数据所有帧导出
- 自动创建目标目录结构
- 标准化的文件命名
- 进度显示和错误处理
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import warnings

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入ACDC数据集类
from acdc_dataset import ACDCDataset

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning, module="SimpleITK")
warnings.filterwarnings("ignore")

# 设置matplotlib为非交互模式
plt.ioff()


class ACDCFrameExporter:
    """
    ACDC数据集帧导出器

    支持导出所有患者的各个帧为PNG图像
    """

    def __init__(
        self,
        data_root: str,
        target_dir: str,
        mode: str = '4d_sequence',  # 使用4D模式导出所有帧
        normalize: bool = True,
        create_subdirs: bool = True
    ):
        """
        初始化导出器

        Args:
            data_root: ACDC数据集根目录
            target_dir: 目标保存目录
            mode: 数据加载模式 ('3d_keyframes', '4d_sequence', 'ed_only', 'es_only')
            normalize: 是否强度归一化
            create_subdirs: 是否为每个患者创建子目录
        """
        self.data_root = Path(data_root)
        self.target_dir = Path(target_dir)
        self.mode = mode
        self.normalize = normalize
        self.create_subdirs = create_subdirs

        # 创建目标目录
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # 初始化数据集
        self._init_dataset()

        print("✅ ACDC帧导出器初始化完成")
        print(f"📁 数据源: {self.data_root}")
        print(f"🎯 目标目录: {self.target_dir}")
        print(f"📊 加载模式: {self.mode}")
        print(f"👥 患者数量: {len(self.dataset)}")

    def _init_dataset(self):
        """初始化数据集"""
        # 尝试训练集和测试集
        for split in ['training', 'testing']:
            try:
                dataset = ACDCDataset(
                    data_root=str(self.data_root),
                    split=split,
                    mode=self.mode,
                    load_segmentation=False,  # 只导出图像，不需要分割
                    normalize=self.normalize,
                    cache_data=False  # 不缓存以节省内存
                )
                self.dataset = dataset
                self.split = split
                break
            except Exception as e:
                print(f"⚠️ {split}数据集加载失败: {e}")
                continue
        else:
            raise RuntimeError("无法加载任何数据集分割")

    def _get_patient_dir(self, patient_id: str) -> Path:
        """获取患者的目标目录"""
        if self.create_subdirs:
            return self.target_dir / patient_id
        else:
            return self.target_dir

    def _get_frame_filename(self, patient_id: str, frame_idx: int, phase: str = None) -> str:
        """
        生成帧文件名

        命名格式: {patient_id}_{phase}_frame{frame_idx:03d}.png
        例如: patient001_ED_frame001.png
        """
        if phase:
            return f"{patient_id}_{phase}_frame{frame_idx:03d}.png"
        else:
            return f"{patient_id}_frame{frame_idx:03d}.png"

    def _save_frame_as_png(self, image: np.ndarray, filepath: Path, title: str = None):
        """
        将图像帧保存为PNG

        Args:
            image: 图像数组 (H, W)
            filepath: 保存路径
            title: 图像标题
        """
        # 确保图像是2D的
        if image.ndim == 3:
            # 如果是3D，选择中间切片
            image = image[image.shape[0] // 2]

        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # 显示图像
        im = ax.imshow(image, cmap='gray')

        # 设置标题
        if title:
            ax.set_title(title, fontsize=12, fontweight='bold')

        ax.axis('off')

        # 保存图像
        plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    def export_patient_frames(self, patient_idx: int, verbose: bool = True) -> int:
        """
        导出单个患者的帧

        Args:
            patient_idx: 患者索引
            verbose: 是否显示详细信息

        Returns:
            导出的帧数量
        """
        try:
            # 获取患者数据
            sample = self.dataset[patient_idx]
            patient_id = sample['patient_id']

            if verbose:
                print(f"🔄 处理患者: {patient_id}")

            # 创建患者目录
            patient_dir = self._get_patient_dir(patient_id)
            patient_dir.mkdir(parents=True, exist_ok=True)

            frames_saved = 0

            # 根据模式处理不同类型的数据
            if self.mode == '4d_sequence':
                # 导出4D序列的所有帧
                if 'image' in sample and sample['image'] is not None:
                    image_4d = sample['image'].numpy() if isinstance(sample['image'], torch.Tensor) else sample['image']

                    # image_4d 形状: (T, Z, H, W)
                    n_frames, n_slices, height, width = image_4d.shape

                    if verbose:
                        print(f"  📹 4D序列: {n_frames}帧 × {n_slices}切片")

                    # 为每个时间帧选择中间切片并保存
                    middle_slice = n_slices // 2
                    for frame_idx in range(n_frames):
                        frame_image = image_4d[frame_idx, middle_slice]

                        filename = self._get_frame_filename(patient_id, frame_idx + 1)  # 帧号从1开始
                        filepath = patient_dir / filename

                        title = f"{patient_id} - Frame {frame_idx + 1:03d}"

                        self._save_frame_as_png(frame_image, filepath, title)
                        frames_saved += 1

            elif self.mode == '3d_keyframes':
                # 导出ED和ES关键帧
                if 'images' in sample and sample['images'] is not None:
                    images = sample['images'].numpy() if isinstance(sample['images'], torch.Tensor) else sample['images']

                    # images 形状: (2, Z, H, W) - ED和ES
                    phases = ['ED', 'ES']
                    n_slices = images.shape[1]

                    if verbose:
                        print(f"  🎯 关键帧: ED + ES, 每帧{n_slices}切片")

                    for phase_idx, phase in enumerate(phases):
                        # 为每个时相的所有切片创建图像
                        for slice_idx in range(n_slices):
                            frame_image = images[phase_idx, slice_idx]

                            filename = self._get_frame_filename(patient_id, slice_idx + 1, phase)
                            filepath = patient_dir / filename

                            title = f"{patient_id} - {phase} - Slice {slice_idx + 1:02d}"

                            self._save_frame_as_png(frame_image, filepath, title)
                            frames_saved += 1

            elif self.mode in ['ed_only', 'es_only']:
                # 导出单个时相的所有切片
                phase = self.mode.split('_')[0].upper()

                if 'image' in sample and sample['image'] is not None:
                    image = sample['image'].numpy() if isinstance(sample['image'], torch.Tensor) else sample['image']

                    # image 形状: (Z, H, W)
                    n_slices = image.shape[0]

                    if verbose:
                        print(f"  🎯 {phase}帧: {n_slices}切片")

                    for slice_idx in range(n_slices):
                        frame_image = image[slice_idx]

                        filename = self._get_frame_filename(patient_id, slice_idx + 1, phase)
                        filepath = patient_dir / filename

                        title = f"{patient_id} - {phase} - Slice {slice_idx + 1:02d}"

                        self._save_frame_as_png(frame_image, filepath, title)
                        frames_saved += 1

            if verbose:
                print(f"  ✅ 已保存 {frames_saved} 帧到 {patient_dir}")

            return frames_saved

        except Exception as e:
            print(f"  ❌ 导出患者 {self.dataset.patient_list[patient_idx]} 失败: {e}")
            return 0

    def export_all_frames(self, start_idx: int = 0, end_idx: int = None, verbose: bool = True) -> dict:
        """
        导出所有患者的帧

        Args:
            start_idx: 开始患者索引
            end_idx: 结束患者索引 (None表示到最后)
            verbose: 是否显示详细信息

        Returns:
            导出统计信息
        """
        if end_idx is None:
            end_idx = len(self.dataset)

        total_patients = end_idx - start_idx
        total_frames = 0
        successful_patients = 0

        print("🚀 开始批量导出帧")
        print(f"👥 处理患者: {start_idx + 1} - {end_idx} (共 {total_patients} 例)")
        print(f"📊 模式: {self.mode}")
        print("-" * 60)

        start_time = datetime.now()

        for i in range(start_idx, end_idx):
            patient_start_time = datetime.now()

            # 导出患者帧
            frames_saved = self.export_patient_frames(i, verbose=verbose)
            total_frames += frames_saved

            if frames_saved > 0:
                successful_patients += 1

            # 显示进度
            patient_time = (datetime.now() - patient_start_time).total_seconds()
            progress = (i - start_idx + 1) / total_patients
            progress_bar = self._create_progress_bar(progress)

            eta = (datetime.now() - start_time).total_seconds() / (i - start_idx + 1) * (total_patients - (i - start_idx + 1))

            print(f"{progress_bar} {i - start_idx + 1:3d}/{total_patients:3d} | "
                  f"患者: {self.dataset.patient_list[i]} | "
                  f"帧数: {frames_saved:3d} | "
                  f"耗时: {patient_time:.1f}s | "
                  f"预计剩余: {eta/60:.1f}min")

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # 导出统计
        stats = {
            'total_patients': total_patients,
            'successful_patients': successful_patients,
            'total_frames': total_frames,
            'total_time_seconds': total_time,
            'average_time_per_patient': total_time / total_patients if total_patients > 0 else 0,
            'frames_per_second': total_frames / total_time if total_time > 0 else 0,
            'export_mode': self.mode,
            'target_directory': str(self.target_dir),
            'export_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        print("\n" + "=" * 60)
        print("🎉 导出完成!")
        print(f"👥 总患者数: {stats['total_patients']}")
        print(f"✅ 成功患者: {stats['successful_patients']}")
        print(f"🖼️ 总帧数: {stats['total_frames']}")
        print(f"⏱️ 总耗时: {stats['total_time_seconds']/60:.1f}分钟")
        print(f"📊 平均每患者耗时: {stats['average_time_per_patient']:.1f}秒")
        print(f"🚀 处理速度: {stats['frames_per_second']:.1f}帧/秒")
        print(f"📁 保存目录: {stats['target_directory']}")
        print("=" * 60)

        return stats

    def _create_progress_bar(self, progress: float, width: int = 30) -> str:
        """创建进度条"""
        filled = int(width * progress)
        bar = '█' * filled + '░' * (width - filled)
        percentage = progress * 100
        return f"[{bar}] {percentage:5.1f}%"

    def get_export_summary(self, stats: dict) -> str:
        """生成导出摘要"""
        summary = f"""
# ACDC数据集帧导出报告

## 导出概况
- **导出时间**: {stats['export_timestamp']}
- **数据模式**: {stats['export_mode']}
- **总患者数**: {stats['total_patients']}
- **成功患者**: {stats['successful_patients']}
- **总帧数**: {stats['total_frames']}

## 性能统计
- **总耗时**: {stats['total_time_seconds']/60:.1f} 分钟
- **平均每患者耗时**: {stats['average_time_per_patient']:.1f} 秒
- **处理速度**: {stats['frames_per_second']:.1f} 帧/秒

## 输出位置
- **目标目录**: {stats['target_directory']}
- **目录结构**: {'每患者子目录' if self.create_subdirs else '统一目录'}

## 文件命名规则
- **4D序列**: `{{patient_id}}_frame{{frame_idx:03d}}.png`
- **关键帧**: `{{patient_id}}_{{phase}}_frame{{slice_idx:03d}}.png`
- **分辨率**: 300 DPI PNG格式

---
*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
        """.strip()

        return summary


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='批量导出ACDC数据集帧为PNG图像')
    parser.add_argument('--data_root', type=str, default='../acdc_dataset',
                       help='ACDC数据集根目录')
    parser.add_argument('--target_dir', type=str, default='./acdc_img_datasets',
                       help='目标保存目录')
    parser.add_argument('--mode', type=str, default='4d_sequence',
                       choices=['3d_keyframes', '4d_sequence', 'ed_only', 'es_only'],
                       help='数据加载模式')
    parser.add_argument('--start_idx', type=int, default=0,
                       help='开始患者索引')
    parser.add_argument('--end_idx', type=int, default=None,
                       help='结束患者索引 (None表示全部)')
    parser.add_argument('--no_subdirs', action='store_true',
                       help='不创建患者子目录')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')

    args = parser.parse_args()

    # 确保路径是绝对路径
    script_dir = Path(__file__).parent
    data_root = script_dir / args.data_root
    target_dir = script_dir / args.target_dir

    print("🖼️ ACDC数据集帧导出工具")
    print("=" * 50)
    print(f"📁 数据目录: {data_root}")
    print(f"🎯 目标目录: {target_dir}")
    print(f"📊 导出模式: {args.mode}")
    print(f"👥 患者范围: {args.start_idx} - {args.end_idx or '全部'}")
    print(f"📂 子目录: {'否' if args.no_subdirs else '是'}")
    print()

    try:
        # 创建导出器
        exporter = ACDCFrameExporter(
            data_root=str(data_root),
            target_dir=str(target_dir),
            mode=args.mode,
            create_subdirs=not args.no_subdirs
        )

        # 执行导出
        stats = exporter.export_all_frames(
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            verbose=args.verbose
        )

        # 生成并保存摘要报告
        summary = exporter.get_export_summary(stats)
        summary_file = Path(target_dir) / "export_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"📝 导出摘要已保存: {summary_file}")

        print("\n🎉 所有任务完成!")
        return 0

    except Exception as e:
        print(f"\n❌ 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
