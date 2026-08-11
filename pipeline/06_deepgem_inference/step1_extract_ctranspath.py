"""Extract CTransPath 768-dim features per patch → per-case .pkl.

参考 DeepGEM/data_prepare/step3_extract_feature.py 的 ConvStem 实现。
加载 model.safetensors (HuggingFace timm-saved format, 27.77M params)。

输入/输出与 step3 同名, 改写成直接遍历 patches_direct/。
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torchvision.transforms as transforms
from PIL import Image
from safetensors.torch import load_file
from timm.models.layers import to_2tuple


# ---- ConvStem (来自 DeepGEM/data_prepare/step3_extract_feature.py) ----
class ConvStem(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768,
                 norm_layer=None, flatten=True):
        super().__init__()
        assert patch_size == 4
        assert embed_dim % 8 == 0
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for _ in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1]
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


def load_ctranspath(ckpt_path: Path, device: str):
    """加载 CTransPath + 注入 ConvStem + load model.safetensors."""
    # timm 0.5: 不支持 embed_layer 参数, 手动建 model 后注入 patch_embed
    # 先用 timm.create_model 直接拿到 swin_tiny, 然后 patch_embed = ConvStem
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False)
    model.patch_embed = ConvStem(embed_dim=model.embed_dim)
    model.head = nn.Identity()
    # Load safetensors
    sd = load_file(str(ckpt_path))
    # Drop the 2 patch_embed.norm.* artifacts (timm 0.5 default PatchEmbed norm, unused)
    sd_clean = {k: v for k, v in sd.items() if not k.startswith('patch_embed.norm.')}
    miss, unex = model.load_state_dict(sd_clean, strict=False)
    sig_miss = [k for k in miss if all(x not in k for x in ['relative_position', 'attn_mask', 'index'])]
    sig_unex = [k for k in unex if all(x not in k for x in ['relative_position', 'attn_mask', 'index'])]
    if sig_miss or sig_unex:
        print(f"[WARN] missing={sig_miss[:3]} unex={sig_unex[:3]}", file=sys.stderr)
    model.eval().to(device)
    print(f"[OK] CTransPath loaded: missing={len(sig_miss)}, unexpected={len(sig_unex)}")
    return model


class PatchDS(data_utils.Dataset):
    def __init__(self, paths, trans):
        self.paths = paths
        self.trans = trans
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256), (0, 0, 0))
        return self.trans(img), str(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=Path, default=Path("./data/TCGA-LUAD-WSI/patches_direct"))
    ap.add_argument("--ckpt", type=Path, default=Path("./DeepGEM/checkpoints/pretrain/model.safetensors"))
    ap.add_argument("--out", type=Path, default=Path("./deepgem_feat"))
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[INFO] device={device}  patches={args.patches}")

    if not args.ckpt.exists():
        sys.exit(f"[FATAL] CTransPath 权重不存在: {args.ckpt}")
    model = load_ctranspath(args.ckpt, device)
    args.out.mkdir(parents=True, exist_ok=True)

    # Collect patch paths
    items = []
    for case_dir in sorted(args.patches.iterdir()):
        if not case_dir.is_dir() or case_dir.name.startswith("_"):
            continue
        for jp in sorted(case_dir.rglob("*.jpg")):
            items.append((case_dir.name, jp))
    if not items:
        sys.exit(f"[FATAL] no jpg under {args.patches}")
    print(f"[INFO] {len(items)} patches  cases={len({c for c,_ in items})}")

    # Skip already-existing pkl (resume)
    skipped = 0
    items_new = []
    for case_id, jp in items:
        rel = jp.relative_to(args.patches / case_id)
        out_path = args.out / case_id / rel.with_suffix(".pkl")
        if out_path.exists():
            skipped += 1
            continue
        items_new.append((case_id, jp))
    items = items_new
    print(f"[INFO] after skip-existing: {len(items)} patches  skipped={skipped}")

    trans = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    ds = PatchDS([p for _, p in items], trans)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, num_workers=args.workers,
        shuffle=False, drop_last=False, pin_memory=(device == "cuda"),
    )

    writer = ThreadPoolExecutor(max_workers=4)
    path_to_case = {str(p): c for c, p in items}

    def save_one(case_id, jp, feat):
        rel = jp.relative_to(args.patches / case_id)
        out_path = args.out / case_id / rel.with_suffix(".pkl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # DeepGEM 期望 list[dict{val, feat_name}]
        payload = [{"val": feat, "feat_name": jp.stem}]
        with open(out_path, "wb") as f:
            pickle.dump(payload, f)

    t0 = time.time(); n = 0
    with torch.no_grad():
        for batch_imgs, batch_paths in dl:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            feat = model(batch_imgs)  # [B, 768]
            feat_np = feat.cpu().numpy().astype(np.float32)
            for fp, fv in zip(batch_paths, feat_np):
                case_id = path_to_case[fp]
                jp = Path(fp)
                writer.submit(save_one, case_id, jp, fv)
            n += len(batch_paths)
            if n % (args.batch_size * 5) == 0:
                rate = n / (time.time() - t0 + 1e-9)
                eta = (len(items) - n) / (rate + 1e-9)
                print(f"  {n}/{len(items)} {rate:.1f} img/s ETA {eta:.0f}s", flush=True)
    writer.shutdown(wait=True)
    dt = time.time() - t0
    print(f"[OK] {n} patches extracted in {dt:.1f}s ({n/dt:.1f} img/s)")


if __name__ == "__main__":
    main()
