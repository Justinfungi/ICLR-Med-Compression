# ACDC数据集帧导出报告

## 导出概况
- **导出时间**: 2025-10-18 22:52:35
- **数据模式**: 3d_keyframes
- **总患者数**: 3
- **成功患者**: 3
- **总帧数**: 60

## 性能统计
- **总耗时**: 0.2 分钟
- **平均每患者耗时**: 3.4 秒
- **处理速度**: 5.9 帧/秒

## 输出位置
- **目标目录**: /home/fish/Documents/ICLR-Med/MedCompression/dataloader/acdc_img_datasets
- **目录结构**: 每患者子目录

## 文件命名规则
- **4D序列**: `{patient_id}_frame{frame_idx:03d}.png`
- **关键帧**: `{patient_id}_{phase}_frame{slice_idx:03d}.png`
- **分辨率**: 300 DPI PNG格式

---
*自动生成于 2025-10-18 22:52:35*