"""All-in-one DeepGEM demo runner for 2 .kfb files (no PNG explosion).

Strategy:
- Monkey-patch openslide.OpenSlide to dispatch .kfb to kfbslide.
- Stream level-0 tiles @ 1120x1120 → resize 224x224 → feed CTransPath → save per-patch features.
- Skip step1/step2: avoid writing 6000 PNGs.
- Skip step4: roll feature-per-patch into one .pt per WSI.
- Then run main/test.py with the merged .pt directly.

This covers the same pipeline as the official DeepGEM, only short-cutting the on-disk
patch stage because level-0 kfb tiles are produced on-demand by kfbslide.
"""
import sys, os, time, pickle, argparse
from pathlib import Path

import kfbslide
import openslide

# ---- Monkey-patch openslide -> kfbslide for .kfb ----
_orig_OpenSlide = openslide.OpenSlide
_orig_open_slide = openslide.open_slide


def _patched_OpenSlide(path):
    s = str(path)
    if s.lower().endswith('.kfb'):
        return kfbslide.open_slide(s)
    return _orig_OpenSlide(s)


def _patched_open_slide(path):
    if str(path).lower().endswith('.kfb'):
        return kfbslide.open_slide(path)
    return _orig_open_slide(path)


openslide.OpenSlide = _patched_OpenSlide
openslide.open_slide = _patched_open_slide
sys.modules['openslide'].OpenSlide = _patched_OpenSlide
sys.modules['openslide'].open_slide = _patched_open_slide

# Now we can import step3 (it doesn't use openslide directly but uses step3.configs)
sys.path.insert(0, str(Path(__file__).parent.parent / "DeepGEM"))
sys.path.insert(0, str(Path(__file__).parent.parent / "DeepGEM" / "data_prepare"))

import torch
import numpy as np
import cv2


def read_patch_from_wsi(wsi, x, y, patch_size, down_scale):
    """Read patch from level 0 and resize to 224x224."""
    big = wsi.read_region(
        (x * down_scale, y * down_scale),
        0,
        (patch_size * down_scale, patch_size * down_scale),
    )
    big = np.array(big)[..., :3]  # drop alpha
    small = cv2.resize(big, (224, 224), interpolation=cv2.INTER_LINEAR)
    return small


def extract_features_for_wsi(wsi, model, device, patch_size=1120, step_size=1120,
                             scale=20, down_scale=1, skip_large_bg=True):
    """Stream through WSI at given scale, extract 768-dim CTransPath features."""
    mpp_x = float(wsi.properties.get("openslide.mpp-x", 0.5))
    auto_down = int(round(40 / scale)) if mpp_x < 0.3 else int(round(20 / scale))
    if auto_down != down_scale:
        print(f"  auto-down override: mpp_x={mpp_x}, using down_scale={auto_down} (scale={scale})")
        down_scale = auto_down

    W, H = wsi.dimensions
    step_x_max = int(np.floor(W / (step_size * down_scale)))
    step_y_max = int(np.floor(H / (step_size * down_scale)))
    total = step_x_max * step_y_max
    print(f"  W={W}, H={H}, mpp_x={mpp_x}, down_scale={down_scale}, total patches: {total}")
    if total == 0:
        print("  WARN: 0 patches, fall back to using full-slide level-0 read at smaller patch")
        return None

    coords = []
    feats = []
    n_skipped = 0
    t0 = time.time()

    # Use TransPath model.eval(). Forward 1 image at a time (safer than batching on tiny inputs)
    # but still keep CPU light.
    with torch.no_grad():
        for j in range(step_y_max):
            for i in range(step_x_max):
                x = i * step_size
                y = j * step_size
                # Read+resize
                small = read_patch_from_wsi(wsi, x, y, patch_size, down_scale)
                # Skip near-white background
                if skip_large_bg:
                    if small.mean() > 220:
                        n_skipped += 1
                        continue
                # To tensor (CTransPath uses ImageNet normalization)
                from torchvision import transforms
                tx = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                t_img = tx(small).unsqueeze(0).to(device)
                feat = model(t_img).cpu().numpy()[0]
                coords.append([x, y])
                feats.append(feat)
            done = (j + 1) * step_x_max
            if (j + 1) % 5 == 0 or done == total:
                print(f"    [{done}/{total}] elapsed {time.time()-t0:.1f}s ({n_skipped} bg-skipped)")
    return np.asarray(feats, dtype=np.float32), np.asarray(coords, dtype=np.int32)


def load_ctranspath(device):
    """Instantiate CTransPath with the bundled checkpoint."""
    import timm
    cands = [
        r"D:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\DeepGEM\ctranspath.pth",
        Path(__file__).parent.parent / "DeepGEM" / "ctranspath.pth",
    ]
    for c in cands:
        if Path(c).exists():
            model_path = str(c)
            break
    else:
        raise FileNotFoundError("ctranspath.pth not found")

    from data_prepare.step3_extract_feature import CTransPath
    model = CTransPath(modelname="swin_tiny_patch4_window7_224", model_path=model_path).to(device)
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True, help="DeepGEM .pth gen head checkpoint")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    here = Path(__file__).parent.parent
    SAMPLE = here / "Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision" / "sample"
    OUT = Path(args.out_dir) if args.out_dir else here / "DeepGEM" / "demo_runs_kfb"
    OUT.mkdir(parents=True, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading CTransPath...")
    backbone = load_ctranspath(DEVICE)
    print(f"Loading gen-head checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    print(f"  ckpt keys: {list(ckpt.keys())}")

    for kfb in sorted(SAMPLE.glob("*.kfb")):
        print(f"\n========== {kfb.name} ==========")
        # kfbslide.OpenSlide — but the monkey patch makes openslide.OpenSlide route to kfbslide
        wsi = openslide.OpenSlide(str(kfb))
        print(f"  mpp_x={wsi.properties.get('openslide.mpp-x')}, "
              f"objective={wsi.properties.get('openslide.objective-power')}")
        result = extract_features_for_wsi(wsi, backbone, DEVICE,
                                           patch_size=1120, step_size=1120, scale=20)
        if result is None:
            continue
        feats, coords = result
        np.savez_compressed(OUT / f"{kfb.stem}_feats.npz", feats=feats, coords=coords)
        print(f"  saved {OUT / f'{kfb.stem}_feats.npz'}  feats={feats.shape}")


if __name__ == "__main__":
    main()
