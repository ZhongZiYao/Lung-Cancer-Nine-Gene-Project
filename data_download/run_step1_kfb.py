"""Run DeepGEM step1_WSI_cropping on .kfb files via kfbslide fall-back for openslide.

Strategy:
- import openslide -> replace its OpenSlide/open_slide with kfbslide equivalents
- import sys.modules['openslide'] -> also patch
- import step1_WSI_cropping's logic via runpy -> it'll use the patched openslide
"""
import sys
import argparse
from pathlib import Path

# Monkey-patch BEFORE we import anything in DeepGEM
import kfbslide


def _patched_open_slide(path):
    p = str(path).lower()
    if p.endswith('.kfb'):
        return kfbslide.open_slide(path)
    import openslide
    return openslide.OpenSlide(path)


import openslide  # original module
openslide.open_slide = _patched_open_slide
# OpenSlide class: we can't subclass because kfbslide's OpenSlide isn't a real
# OpenSlide instance. But step1 only calls OpenSlide(path) — so returning a
# kfbslide handle works.

# Wrap kfbslide's OpenSlide as a class-callable shim. The class itself returns
# a kfbslide handle if path is .kfb, else a real OpenSlide.
_orig_OpenSlide = openslide.OpenSlide


def _patched_OpenSlide(path):
    s = str(path)
    if s.lower().endswith('.kfb'):
        return kfbslide.open_slide(s)
    return _orig_OpenSlide(path)


openslide.OpenSlide = _patched_OpenSlide

# Make `import openslide as slide` use the patched class
sys.modules['openslide'].OpenSlide = _patched_OpenSlide
sys.modules['openslide'].open_slide = _patched_open_slide

# Now safe-import step1
sys.path.insert(0, str(Path(__file__).parent.parent / "DeepGEM" / "data_prepare"))
import step1_WSI_cropping as step1

if __name__ == "__main__":
    # Defaults that match our 2 kfb files
    here = Path(__file__).parent.parent   # data_download/.. -> project root
    WSI_DIR = here / "Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision" / "sample"
    OUTPUT = here / "DeepGEM" / "data" / "patch_kfb_demo"

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default=str(WSI_DIR))
    p.add_argument("--output", type=str, default=str(OUTPUT))
    p.add_argument("--patch_size", type=int, default=1120)
    p.add_argument("--scale", type=int, default=20)
    p.add_argument("--num_threads", type=int, default=2)
    p.add_argument("--overlap", type=int, default=0)
    args = p.parse_args()

    import glob, numpy as np, threading
    from tqdm import tqdm
    from os.path import join, isdir
    from os import makedirs

    all_slides = (glob.glob(f"{args.dataset}/*.kfb")
                  + glob.glob(f"{args.dataset}/*.svs")
                  + glob.glob(f"{args.dataset}/*.tif")
                  + glob.glob(f"{args.dataset}/*.tiff"))
    print(f"Found {len(all_slides)} slides: {all_slides}")

    patch_size = args.patch_size
    overlap = args.overlap
    scale = args.scale
    step = patch_size - overlap
    out_base = args.output

    each = int(np.floor(len(all_slides) / args.num_threads))
    threads = []
    for i in range(args.num_threads):
        if i < (args.num_threads - 1):
            t = threading.Thread(target=step1.slide_to_patch,
                                 args=(out_base, all_slides[each * i: each * (i + 1)],
                                       patch_size, step, scale))
        else:
            t = threading.Thread(target=step1.slide_to_patch,
                                 args=(out_base, all_slides[each * i:], patch_size, step, scale))
        threads.append(t)
    for t in threads: t.start()
    for t in threads: t.join()

    # build dict_name2imgs.pkl like step1 does
    import pickle
    out_list = sorted(Path(out_base).iterdir()) if isdir(out_base) else []
    d = {}
    for case in out_list:
        if case.is_dir():
            pngs = sorted(case.glob("*.png"))
            d[case.name] = [str(p) for p in pngs]
    with open(Path(out_base) / "dict_name2imgs.pkl", "wb") as f:
        pickle.dump(d, f)
    print(f"\nDone. Wrote {len(d)} slides' patch lists. Total patches: "
          f"{sum(len(v) for v in d.values())}")
