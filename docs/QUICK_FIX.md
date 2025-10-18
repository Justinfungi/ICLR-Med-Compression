# 🚀 快速修复指南 - MRI图像适配TiTok

## ❌ 问题1: 通道数不匹配

```
RuntimeError: expected input[1, 4, 2370, 1950] to have 3 channels, but got 4 channels instead
```

**原因**: PNG保存的MRI图像是RGBA(4通道),但TiTok需要RGB(3通道)

## ❌ 问题2: 分辨率不匹配

```
RuntimeError: The size of tensor a (17909) must match the size of tensor b (257) at non-singleton dimension 1
```

**原因**: 图像分辨率2370×1950太大,TiTok训练在256×256分辨率上

## ✅ 完整解决方案

需要同时解决两个问题:
1. **RGBA → RGB** (4通道→3通道)
2. **任意分辨率 → 256×256** (调整大小)

## 🔧 快速修复代码

在 `demo.py` 的 `tokenize_and_reconstruct` 函数中添加:

```python
def tokenize_and_reconstruct(img_path, titok_tokenizer, device):
    original_image = Image.open(img_path)

    # ✅ 修复1: RGBA → RGB
    if original_image.mode == 'RGBA':
        rgb_image = Image.new('RGB', original_image.size, (255, 255, 255))
        rgb_image.paste(original_image, mask=original_image.split()[3])
        original_image = rgb_image
    elif original_image.mode != 'RGB':
        original_image = original_image.convert('RGB')

    # ✅ 修复2: 调整到256×256
    target_size = (256, 256)
    if original_image.size != target_size:
        original_image = original_image.resize(target_size, Image.Resampling.LANCZOS)

    # 现在可以正常处理了
    image = torch.from_numpy(np.array(original_image).astype(np.float32))
    image = image.permute(2, 0, 1).unsqueeze(0) / 255.0  # [1, 3, 256, 256] ✅
    # 后续代码不变...
```

## 📝 完整新文件

我已创建:
- `/root/Documents/ICLR-Med/1d-tokenizer/demo_mri.py` - 完整修复版本
- `/root/Documents/ICLR-Med/MedCompression/docs/README.md` - 详细说明

## 🎯 运行修复后的代码

```bash
cd /root/Documents/ICLR-Med/1d-tokenizer
python demo_mri.py
```

## 📊 问题根源详解

### 问题1: 通道数

| 组件 | 期望 | 实际 | 问题 |
|------|------|------|------|
| **TiTok输入** | RGB (3通道) | RGBA (4通道) | ❌ 不匹配 |
| **Conv层** | `Conv2d(3, 1024, ...)` | 收到4通道输入 | ❌ 维度错误 |
| **原因** | matplotlib保存PNG | 默认带Alpha通道 | ⚠️ |

### 问题2: 分辨率

| 组件 | 期望 | 实际 | 问题 |
|------|------|------|------|
| **TiTok输入** | 256×256 | 2370×1950 | ❌ 不匹配 |
| **位置编码** | 257个位置 (16×16+1) | 17909个位置 | ❌ 维度错误 |
| **Patch数** | 16×16=256 patches | 148×121=17908 patches | ❌ 超出范围 |
| **原因** | ImageNet训练分辨率 | MRI高分辨率保存 | ⚠️ |

**计算说明**:
```python
# TiTok使用16×16的patch size
256 ÷ 16 = 16  # 每边16个patches
16 × 16 = 256  # 总共256个patches
256 + 1 = 257  # 加上class token

# 你的MRI图像
2370 ÷ 16 = 148.125 → 148 patches
1950 ÷ 16 = 121.875 → 121 patches
148 × 121 = 17908 patches
17908 + 1 = 17909  # ❌ 远超257!
```

## 🎨 形状变化

```python
# 保存的PNG文件
PIL.Image.open("mri.png")
├─ mode: RGBA
└─ array.shape: (H, W, 4)  # ❌ 4通道

# 转换为RGB
image.convert('RGB')
├─ mode: RGB
└─ array.shape: (H, W, 3)  # ✅ 3通道

# PyTorch tensor
torch.tensor(array).permute(2,0,1).unsqueeze(0)
└─ shape: [1, 3, H, W]  # ✅ 正确!
```

## 💡 其他解决方案

### 方案A: 保存时避免Alpha通道
```python
# 在example_usage.py中
fig.savefig(path, facecolor='white')  # 强制白色背景
```

### 方案B: 转换所有通道
```python
original_image = original_image.convert('RGB')  # 万能转换
```

### 方案C: 直接处理numpy数组
```python
# 跳过PNG,直接从dataset处理
mri_slice = dataset[0]['images'][0, 5].numpy()  # (256, 216)
rgb = np.stack([mri_slice]*3, axis=0)  # (3, 256, 216)
```

---

**问题**: 4通道RGBA vs 3通道RGB
**原因**: PNG保存格式
**修复**: `image.convert('RGB')`
**文件**: `demo_mri.py` (已创建)
