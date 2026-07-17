"""
step1_extract_features_uni.py
──────────────────────────────
从 extract_patches_direct.py 的输出(patches_direct/<case>/<slice>/<x>_<y>.jpg),
用 UNI 抽 1024-dim 特征,存成 .npz

输入:
  patches_dir/  .../<case>/<slice_uid>/<x>_<y>.jpg
输出:
  features/<case>.npz
    {
      "features":   [N_total_patches, 1024] float32
      "slice_ids":  [N_total_patches]     int     (每 patch 来自哪个 slice)
      "slice_uids": [n_slices]            str     (slice_uid list)
      "patch_paths":[N_total_patches]     str     (patch 相对路径)
    }
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import timm
from timm.data import resolve_data_config


# ──────────────────────────────────────────────
# 1. UNI 模型包装 (Vit-L/16, 1024-dim)
# ──────────────────────────────────────────────
class UNIWrapper(torch.nn.Module):
    """把 UNI 的 forward 封装成 timm-style,只返回 [B, 1024] CLS feature"""

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        super().__init__()
        # UNI 暴露的接口(参考 uni/get_encoder/get_encoder.py)
        # 它内部用 timm.create_model("vit_large_patch16_224", ...)
        # 我们直接新建同样骨架,加载它的 state_dict
        self.model = timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=0,  # 不要 head,只要 1024-dim CLS feature
        )
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # state_dict 可能带前缀 "model." 或直接是 backbone keys, 兼容两种
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"[UNI] loaded, missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
        self.model.eval()
        self.device = device
        self.model.to(device)

        cfg = resolve_data_config({}, model=self.model)
        self.tf = transforms.Compose([
            transforms.Resize(cfg["input_size"][1:]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg["mean"], std=cfg["std"]
            ),
        ])

    @torch.no_grad()
    def forward(self, x):
        return self.model(x)  # [B, 1024]


# ──────────────────────────────────────────────
# 2. Dataset / DataLoader
# ──────────────────────────────────────────────
class PatchDataset(Dataset):
    def __init__(self, paths: list[str], tf):
        self.paths = paths
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.paths[i]


# ──────────────────────────────────────────────
# 3. 主流程
# ──────────────────────────────────────────────
def extract_one_case(model: UNIWrapper, case_dir: Path, batch_size: int = 32) -> dict:
    # 收集所有 patch (递归 case_dir 找 .jpg)
    patches = sorted(p for p in case_dir.rglob("*.jpg"))
    if not patches:
        return None

    ds = PatchDataset([str(p) for p in patches], model.tf)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)

    feats = []
    kept_paths = []
    for batch_imgs, batch_paths in dl:
        batch_imgs = batch_imgs.to(model.device, non_blocking=True)
        f = model(batch_imgs)
        feats.append(f.cpu().float().numpy())
        kept_paths.extend(batch_paths)
    feats = np.concatenate(feats, axis=0)  # [N, 1024]

    # slice_ids: 从 path 里抠 slice_uid
    # patches/<case>/<slice_uid>/<x>_<y>.jpg
    slice_uids_in_order = []
    slice_id_map = {}
    slice_ids = np.zeros(len(kept_paths), dtype=np.int32)
    for i, p in enumerate(kept_paths):
        slice_uid = Path(p).parent.name
        if slice_uid not in slice_id_map:
            slice_id_map[slice_uid] = len(slice_uids_in_order)
            slice_uids_in_order.append(slice_uid)
        slice_ids[i] = slice_id_map[slice_uid]

    return {
        "features": feats.astype(np.float32),
        "slice_ids": slice_ids,
        "slice_uids": slice_uids_in_order,
        "patch_paths": kept_paths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default="transfer_package/data/TCGA-LUAD-WSI/patches_direct")
    ap.add_argument("--out", default="pipeline/features")
    ap.add_argument("--ckpt", default="UNI/assets/ckpts/uni/pytorch_model.bin")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="指定 case id;不给就跑所有")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    model = UNIWrapper(args.ckpt, device=device)

    patches_root = Path(args.patches)
    if args.cases:
        case_dirs = [patches_root / c for c in args.cases]
    else:
        case_dirs = sorted(d for d in patches_root.iterdir()
                           if d.is_dir() and d.name.startswith("TCGA-"))

    for cd in case_dirs:
        if not cd.is_dir():
            print(f"[skip] {cd} not a dir")
            continue
        print(f"\n[case] {cd.name}")
        d = extract_one_case(model, cd, batch_size=args.batch_size)
        if d is None:
            print(f"  no patches, skip")
            continue
        out_path = out / f"{cd.name}.npz"
        np.savez(
            out_path,
            features=d["features"],
            slice_ids=d["slice_ids"],
            slice_uids=np.array(d["slice_uids"], dtype=object),
            patch_paths=np.array(d["patch_paths"], dtype=object),
        )
        print(f"  -> {out_path}: features={d['features'].shape}, "
              f"slices={len(d['slice_uids'])}, "
              f"total_patches={len(d['patch_paths'])}")


if __name__ == "__main__":
    main()
