"""
ACDC Heart MRI Dataset Loader
专业的ACDC心脏MRI数据集PyTorch加载器

主要组件:
- ACDCDataset: 核心数据集类
- ACDCDataModule: 便捷的数据模块
- 数据变换: 医学图像专用的3D变换
- 分析工具: 数据集统计和可视化
"""

# 核心数据集类
from .acdc_dataset import (
    ACDCDataset,
    ACDCDataModule,
    collate_fn
)

# 从utils导入所有工具
from . import utils
from .utils import *

# 版本信息
__version__ = "1.0.0"
__author__ = "ACDC DataLoader Team"
__email__ = "support@acdcdataloader.com"

# 导出的主要接口 (从utils.__init__.py自动导入)
__all__ = [
    # 数据集
    'ACDCDataset',
    'ACDCDataModule', 
    'collate_fn',
] + utils.__all__  # 添加utils中的所有导出

# 快速使用示例
def quick_start_example():
    """
    快速开始示例代码
    """
    example_code = '''
# 快速开始使用ACDC数据加载器

from dataloader import ACDCDataset, ACDCDataModule, get_train_transforms

# 1. 创建数据集
dataset = ACDCDataset(
    data_root="path/to/acdc_dataset",
    split='training',
    mode='3d_keyframes',
    load_segmentation=True,
    transform=get_train_transforms(output_size=(8, 224, 224))
)

# 2. 使用数据模块
data_module = ACDCDataModule(
    data_root="path/to/acdc_dataset",
    batch_size=4,
    mode='3d_keyframes'
)
data_module.setup()
train_loader = data_module.train_dataloader()

# 3. 获取数据
sample = dataset[0]
print(f"图像形状: {sample['images'].shape}")
print(f"疾病类型: {sample['patient_info']['Group']}")
'''
    return example_code

# 检查依赖
def check_dependencies():
    """检查所需依赖是否已安装"""
    required_packages = [
        'torch',
        'numpy', 
        'matplotlib',
        'SimpleITK',
        'scipy',
        'sklearn',
        'pandas',
        'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️ 缺少以下依赖包: {missing_packages}")
        print(f"请运行: pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ 所有依赖包已安装")
        return True

# 模块导入时的欢迎信息
def _show_welcome():
    """显示欢迎信息"""
    welcome_msg = f"""
🫀 ACDC Heart MRI Dataset Loader v{__version__}
================================================

专业的ACDC心脏MRI数据集PyTorch加载器

主要功能:
✅ 多模式数据加载 (3D/4D, ED/ES, 时序)
✅ 智能数据预处理 (重采样, 归一化)  
✅ 丰富的3D医学图像数据增强
✅ 心脏功能指标自动计算
✅ 数据集统计分析和可视化
✅ 完整的使用示例和文档

快速开始:
>>> from dataloader import ACDCDataset, quick_start_example
>>> print(quick_start_example())

检查依赖:
>>> from dataloader import check_dependencies
>>> check_dependencies()

更多信息请查看 README.md
"""
    print(welcome_msg)

# 只在直接导入时显示欢迎信息
if __name__ != "__main__":
    import os
    if os.getenv("ACDC_LOADER_QUIET") != "1":
        _show_welcome()

# 配置常量
class Config:
    """配置常量"""
    
    # 疾病类型映射
    DISEASE_MAPPING = {
        'NOR': 0,   # 正常
        'DCM': 1,   # 扩张性心肌病
        'HCM': 2,   # 肥厚性心肌病  
        'ARV': 3,   # 异常右心室
        'MINF': 4   # 心梗后改变
    }
    
    # 分割标签映射
    SEGMENTATION_LABELS = {
        'background': 0,      # 背景
        'rv_cavity': 1,       # 右心室腔
        'lv_myocardium': 2,   # 左心室心肌
        'lv_cavity': 3        # 左心室腔
    }
    
    # 默认体素间距
    DEFAULT_SPACING = (1.5625, 1.5625, 10.0)
    
    # 默认图像大小
    DEFAULT_IMAGE_SIZE = (256, 216)
    
    # 支持的加载模式
    SUPPORTED_MODES = [
        '3d_keyframes',
        '4d_sequence', 
        'ed_only',
        'es_only'
    ]

# 暴露配置
__all__.append('Config')
