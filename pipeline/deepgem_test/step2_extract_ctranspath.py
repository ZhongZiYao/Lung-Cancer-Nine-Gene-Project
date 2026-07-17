"""
deepgem_test/step2_extract_ctranspath.py  (final - Plan A + slice-aware)

用 TransPath 官方 TransPath/net/models/modeling.py 的 VisionTransformer
(架构 R50-ViT-B_16 = ResNet50 hybrid stem + ViT-B/16, img_size 256, embed_dim 768)

加载 ctranspath.pth (801 MB) → forward(256x256 patch) → 768-dim CLS feature.

按 SM 子目录切分输出: 每个 case 一个 .npz, 包含:
  features: [sum_N, 768]
  slice_ids: [sum_N]          每个 patch 来自哪个 SM (str)
  slice_uids: [n_slices]      case 的所有 SM uid (sorted)
  patch_paths: [sum_N]        保留作 debug 用
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


HERE = Path(__file__).parent.resolve()
TRANSROOT = HERE.parent.parent / "TransPath"
DEEPGEM_ROOT = HERE.parent.parent / "DeepGEM"
DEFAULT_CTP = DEEPGEM_ROOT / "checkpoints" / "pretrain" / "ctranspath.pth"


def load_ctranspath(ckpt_path, device="cuda"):
    sys.path.insert(0, str(TRANSROOT))
    import net.models.modeling as modeling

    model = modeling.VisionTransformer(
        modeling.CONFIGS['R50-ViT-B_16'],
        img_size=256, zero_head=True, num_classes=1000,
    )
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    new_sd = {k[len("module.net."):]: v for k, v in sd.items()
              if k.startswith("module.net.")}
    msg = model.load_state_dict(new_sd, strict=False)
    print(f"[ctp] loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    model.head = nn.Identity()
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return model, tf


class _GroupedDataset(Dataset):
    """一个 SM 子目录对应一个 sub-bag。path 全部 str, 防 torch collate 炸。"""

    def __init__(self, files, tf):
        if len(files) > 1024:
            idx = np.linspace(0, len(files) - 1, 1024, dtype=int)
            files = [files[i] for i in idx]
        self.files = files
        self.tf = tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), self.files[i]


def extract_one_case(model, tf, case_dir, max_per_slice, batch_size, device):
    """按 SM 子目录分, 每个 SM 一个 sub-bag, 输出 {slice_uid: features}"""
    slice_dirs = sorted([d for d in case_dir.iterdir() if d.is_dir()])
    if not slice_dirs:
        return {}

    slice_results = {}
    for sd in slice_dirs:
        # ⚠️ path 全部 str 避免 collate 炸
        files = sorted(str(p) for p in sd.glob("*.jpg"))
        if not files:
            continue
        ds = _GroupedDataset(files, tf)
        if len(ds) == 0:
            continue
        dl = DataLoader(ds, batch_size=batch_size, num_workers=2, pin_memory=True)
        feats, kept = [], []
        with torch.no_grad():
            for imgs, paths in dl:
                imgs = imgs.to(device, non_blocking=True)
                f = model(imgs)
                feats.append(f.cpu().float().numpy())
                kept.extend(paths)

        slice_feats = np.concatenate(feats, axis=0)
        # sub-sample 到 max_per_slice
        if len(slice_feats) > max_per_slice:
            idx = np.linspace(0, len(slice_feats) - 1, max_per_slice, dtype=int)
            slice_feats = slice_feats[idx]
            kept = [kept[i] for i in idx]
        slice_results[sd.name] = (slice_feats.astype(np.float32), kept)

    return slice_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default=str(HERE / "patches"))
    ap.add_argument("--out", default=str(HERE / "features"))
    ap.add_argument("--ctranspath", default=str(DEFAULT_CTP))
    ap.add_argument("--cases", nargs="*",
                    default=["TCGA-05-4382", "TCGA-05-4249", "TCGA-05-4395"])
    ap.add_argument("--max-per-slice", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    t0 = time.time()
    model, tf = load_ctranspath(Path(args.ctranspath), device=device)
    print(f"[ctp] loaded in {time.time()-t0:.1f}s")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in args.cases:
        case_dir = Path(args.patches) / case
        if not case_dir.exists():
            print(f"[skip] {case_dir}")
            continue

        result = extract_one_case(model, tf, case_dir, args.max_per_slice,
                                   args.batch_size, device)
        if not result:
            print(f"[skip] {case}: no slices")
            continue

        # concat all slice feats, 记录 slice_ids
        feats_list, ids_list, paths_list = [], [], []
        for slice_uid, (slice_feats, kept) in result.items():
            feats_list.append(slice_feats)
            ids_list.extend([str(slice_uid)] * len(slice_feats))
            paths_list.extend(kept)

        feats_concat = np.concatenate(feats_list, axis=0)
        slice_uids_sorted = sorted(result.keys())

        np.savez(out_dir / f"{case}.npz",
                 features=feats_concat,
                 slice_ids=np.array(ids_list, dtype=object),
                 slice_uids=np.array(slice_uids_sorted, dtype=object),
                 patch_paths=np.array(paths_list, dtype=object))
        n_per_slice = " ".join(f"{s[:8]}={len(f)}" for s, (f, _) in result.items())
        print(f"  [{case}] total={feats_concat.shape}, slices=({n_per_slice})")


if __name__ == "__main__":
    main()
