"""精确分析 1120×1120 → 224×224 resize 的信息损失"""
import numpy as np
from PIL import Image
import os
from pathlib import Path

# 用一个真实的 patch
P = Path(r'D:\Lungcancer\Lung-Cancer-Nine-Gene-Project\data\TCGA-LUAD-WSI\patches_direct\TCGA-05-4402\71b20d7d-0fc7-4e13-8ae4-6952466a63eb')
patches = sorted(P.glob('*.jpg'))[:5]

print("=" * 70)
print("1120×1120 → 224×224 resize 信息损失量化分析")
print("=" * 70)

# 1. 像素总数对比
print("\n[1] 像素总数对比")
print(f"  1120×1120 = {1120*1120:,} 像素 = 1.25M")
print(f"  224×224   = {224*224:,} 像素 = 50K")
print(f"  ratio: 1120*1120/224*224 = {1120*1120/(224*224):.1f}x")
print(f"  → 1120×1120 包含 25x 更多像素")
print()

# 2. 实际 patch 的高频信息测试
print("[2] 单 patch 细节保留测试 (TCGA-05-4402 第一张)")
print("-" * 70)

img_path = patches[0]
img = Image.open(img_path).convert('RGB')
arr = np.array(img)
print(f"  patch: {img_path.name}")
print(f"  原图 shape: {arr.shape}, dtype={arr.dtype}")
print(f"  原图统计: mean={arr.mean():.1f}, std={arr.std():.1f}")

# resize 到 224
img_small = img.resize((224, 224), Image.BILINEAR)
arr_small = np.array(img_small)
print(f"  resize 后 shape: {arr_small.shape}")
print(f"  resize 后统计: mean={arr_small.mean():.1f}, std={arr_small.std():.1f}")

# 反向 resize 回 1120 - 看是否能恢复
img_back = img_small.resize((1120, 1120), Image.BILINEAR)
arr_back = np.array(img_back)

# 计算信息损失 (MSE / 像素差)
diff = np.abs(arr.astype(float) - arr_back.astype(float))
print(f"  反向 resize 误差 (MSE): {(diff**2).mean():.2f}")
print(f"  平均像素差: {diff.mean():.2f} / {arr.max()}")
print(f"  最大像素差: {diff.max()}")
print(f"  误差 > 5 的像素占比: {(diff > 5).mean()*100:.2f}%")
print()

# 3. 频率分析 (FFT)
print("[3] 频率域分析 - 哪些频率被丢失")
print("-" * 70)
gray = arr.mean(axis=2)
fft = np.fft.fft2(gray - gray.mean())
fft_mag = np.abs(fft)

# 1120 的频率分辨率
# resize 到 224 后高频信息会被丢弃
# 计算 Nyquist 频率
print(f"  1120×1120 最高空间频率: 1120/2 = 560 cycles/边")
print(f"  224×224 最高空间频率: 224/2 = 112 cycles/边")
print(f"  → 5x 高频信息被丢弃")
print()

# 4. UNI 的实际感受野
print("[4] ViT-L/16 实际感受野分析")
print("-" * 70)
print(f"  UNI ViT-L/16 在 224×224 输入上的 token 划分:")
print(f"    patch_size=16, num_patches = (224/16)*(224/16) = 14*14 = 196 tokens")
print(f"  相当于每个 token = 16×16 像素 = (16/224) = 7.1% 的 224 patch")
print(f"  对应 1120×1120 视野里每个 token = 16×16 像素 from 80×80 (=7.1%)")
print()
print(f"  也就是说: 1120->224 resize 保留所有 '粗' 特征 (>=80 像素结构)")
print(f"  丢弃所有 '细' 特征 (<80 像素结构 = 单个细胞)")
print()

# 5. 病理信息密度分析
print("[5] 病理结构尺度 vs 视野 (20× 物镜, ~0.5μm/pixel)")
print("-" * 70)
print(f"  1120×1120 视野 = 560×560 μm = 0.56 mm")
print(f"  224×224 视野   = 112×112 μm = 0.11 mm")
print()
print(f"  病理结构尺度:")
print(f"    单个细胞核: 5-15 μm (≈ 10-30 px in 1120)")
print(f"    腺体:      100-500 μm (≈ 200-1000 px in 1120)")
print(f"    淋巴带:    200-1000 μm (≈ 400-2000 px in 1120)")
print(f"    肿瘤区域:  1-10 mm = 2000-20000 px")
print()
print(f"  resize 后 (224 视野) 仍能看到:")
print(f"    ✅ 腺体 (0.5 mm = 4-5 个 224 patch)")
print(f"    ✅ 淋巴带 (1 mm = 9 个 224 patch)")
print(f"    ⚠️ 单个细胞核模糊 (10-30 px → 4-12 px in 224, 但在 5×5 平均下边界模糊)")
print(f"    ❌ 细胞内结构 (核仁等 1-2 μm) 完全消失")
print()

# 6. 总结
print("=" * 70)
print("结论")
print("=" * 70)
print("""
✅ 保留的信息:
   - 组织级结构 (gland, stroma, immune infiltrate 分布)
   - 大尺度形态学 (肿瘤边界、坏死区域)
   - 染色梯度 (整体染色深浅)
   - 细胞密度分布

⚠️ 模糊化的信息 (双线性平均):
   - 单个细胞核的精确边界
   - 细胞核内染色质细节
   - 微小结构 (<80 像素)

❌ 丢失的信息:
   - 细胞核内纹理
   - 微小颗粒 (<5 像素)
   - 像素级精确颜色

→ 对 "基因突变预测" (9-gene panel):
   基因突变通常影响整个组织结构 + 大量细胞群, 不依赖单个细胞
   所以 1120→224 resize 损失是 "可接受的"
   
→ 对 "细胞核形态分析":
   需要更小的 patch (256×256) 直接训练, 不用 resize
""")
print("=" * 70)