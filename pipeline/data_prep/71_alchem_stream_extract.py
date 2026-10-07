# -*- coding: utf-8 -*-
"""71_alchem_stream_extract.py - stream ALCHEMIST slides -> Midnight features -> delete slide

Why streaming: slides are ~1.35 GB each, 1,175 cases = 1.59 TB. Disk cannot hold that,
but peak disk stays under ~5 GB if each slide is fetched, consumed and deleted in turn.

Why parallel ranges: GDC's /data endpoint caps one connection at ~1.2 MB/s.
Measured here (2026-10-07):  1 -> 1.2 MB/s | 4 -> 3.6 | 12 -> 6.6 | 24 -> 7.1 MB/s.
12 is the sweet spot, so every file is fetched as 12 concurrent byte ranges.

Tiling: read a few large contiguous blocks (fast on pyramidal TIFF) and cut tiles
inside them. 512 scattered read_region calls are slow random access on AtoB tiles.

Midnight: Dinov2Model, hidden 1536, patch 14, image_size 518, trained at 224px with
mean/std 0.5. We tile 224px at 20x (0.5 mpp), S_INST=512, matching the rest of this
project's patch budget.
"""
import os, sys, time, threading, argparse, traceback
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, pandas as pd, requests, torch
from PIL import Image
import openslide

R = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer"
AL = os.path.join(R, "data", "alchemist")
TMP = os.path.join(AL, "tmp_svs")
GDC = "https://api.gdc.cancer.gov/data/"
GENES = ["KRAS", "TP53", "KEAP1", "NF1", "STK11", "EGFR"]

TILE, TARGET_MPP, N_INST, BLOCK, N_BLOCK, NPAR, BS = 224, 0.5, 512, 2048, 6, 12, 32


def args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--only-luad", type=int, default=0)
    ap.add_argument("--model", default=os.path.join(R, "models", "midnight"))
    ap.add_argument("--tag", default="alchem")
    return ap.parse_args()


def fetch(file_id, size, dst):
    """Download one GDC file as NPAR parallel byte ranges into a pre-sized file."""
    span = max(1, size // NPAR)
    errs = []

    def w(i):
        s0 = i * span
        e0 = size - 1 if i == NPAR - 1 else s0 + span - 1
        if e0 < s0:
            return
        try:
            r = requests.get(GDC + file_id, headers={"Range": "bytes=%d-%d" % (s0, e0)},
                             stream=True, timeout=(30, 180))
            if r.status_code not in (200, 206):
                errs.append("HTTP %d" % r.status_code)
                r.close()
                return
            with open(dst, "r+b") as f:
                f.seek(s0)
                for ch in r.iter_content(1 << 20):
                    if ch:
                        f.write(ch)
            r.close()
        except Exception as e:
            errs.append(str(e)[:80])

    with open(dst, "wb") as f:
        f.truncate(size)
    ths = [threading.Thread(target=w, args=(i,)) for i in range(NPAR)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    if errs:
        raise RuntimeError("download errors: %s" % errs[:3])
    if os.path.getsize(dst) != size:
        raise RuntimeError("size mismatch %d != %d" % (os.path.getsize(dst), size))


def tiles(path, rng):
    """Cut N_INST 224px tiles from a handful of contiguous blocks."""
    osr = openslide.OpenSlide(path)
    w, h = osr.dimensions
    try:
        mpp = float(osr.properties.get(openslide.PROPERTY_NAME_MPP, "nan"))
    except Exception:
        mpp = float("nan")
    ds = 1.0 if not (mpp == mpp and mpp > 0) else max(1.0, mpp / TARGET_MPP)
    px = int(TILE * ds)
    bx = min(int(BLOCK * ds), int((w - px) // 1) if w > px else px)
    by = min(int(BLOCK * ds), int((h - px) // 1) if h > px else px)
    bx = max(px, bx); by = max(px, by)
    out = []
    seen = set()
    for _ in range(N_BLOCK * 10):
        if len(out) >= N_INST:
            break
        x0 = int(rng.randint(0, max(1, w - bx)))
        y0 = int(rng.randint(0, max(1, h - by)))
        key = (x0 // 5000, y0 // 5000)
        if key in seen:
            continue
        seen.add(key)
        img = osr.read_region((x0, y0), 0, (bx, by)).convert("RGB")
        a = np.asarray(img)
        nb = min(a.shape[0], a.shape[1]) // px
        for iy in range(nb):
            for ix in range(nb):
                t = a[iy * px:(iy + 1) * px, ix * px:(ix + 1) * px]
                if t.shape[0] < TILE or t.shape[1] < TILE:
                    continue
                if px != TILE:
                    t = np.asarray(Image.fromarray(t).resize((TILE, TILE), Image.BILINEAR))
                out.append(t)
        img.close()
    osr.close()
    if not out:
        raise RuntimeError("no tiles (level0 %dx%d, ds=%.2f)" % (w, h, ds))
    out = np.stack(out)
    if len(out) > N_INST:
        out = out[rng.choice(len(out), N_INST, replace=False)]
    elif len(out) < N_INST:
        out = np.concatenate([out, out[rng.randint(0, len(out), N_INST - len(out))]])
    return out, dict(mpp=mpp, ds=ds, level0_w=w, level0_h=h, n_available=len(out))


@torch.no_grad()
def embed(model, arr):
    """Return [CLS | mean(patch tokens)] concatenated -> 3072-d per instance.

    CLS is what the Midnight card specifies for classification; mean-pooling over
    patch tokens is the other standard choice for WSI MIL. Storing both (3072-d)
    costs 3.1 MB/slide, which is nothing against a 1.59 TB download, and lets the
    downstream experiment pick one without re-extracting anything.
    """
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float().div_(255.0).sub_(0.5).div_(0.5)
    outs = []
    for i in range(0, len(x), BS):
        o = model(pixel_values=x[i:i + BS].cuda().half())
        h = o.last_hidden_state if hasattr(o, "last_hidden_state") else o[0]
        v = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        outs.append(v.float().cpu())
    return torch.cat(outs).numpy().astype(np.float16)


def main():
    a = args()
    os.makedirs(TMP, exist_ok=True)
    outdir = os.path.join(AL, "features_%s" % a.tag)
    os.makedirs(outdir, exist_ok=True)

    slides = pd.read_csv(os.path.join(AL, "alchem_slides.csv"))
    labs = pd.read_csv(os.path.join(AL, "alchem_labels_6gene.csv"))
    m = slides.merge(labs, on="case_id")
    m["_p"] = m.file_name.str.extract(r"-(S\d+)\.")[0].fillna("S9")
    m = m.sort_values(["case_id", "_p"]).groupby("case_id", as_index=False).first()
    m = m.sort_values("case_id").reset_index(drop=True)
    if a.only_luad:
        m = m[m.diagnosis.str.contains("Adenocarcinoma|Adenosquamous", na=False)]
        m = m.sort_values("case_id").reset_index(drop=True)
    todo = m.iloc[a.start:a.start + a.n]
    print("[job] %d slides (start=%d, only_luad=%d)" % (len(todo), a.start, a.only_luad), flush=True)
    print("[job] out = %s" % outdir, flush=True)

    from transformers import AutoModel
    t0 = time.time()
    model = AutoModel.from_pretrained(a.model, dtype=torch.float16).cuda().eval()
    print("[model] %s loaded in %.0fs, GPU %.1f GB"
          % (os.path.basename(a.model), time.time() - t0,
             torch.cuda.memory_allocated() / 1e9), flush=True)

    feats, index = [], []
    T0 = time.time()
    for k, row in enumerate(todo.itertuples()):
        dst = os.path.join(TMP, "%s.svs" % row.file_id)
        try:
            td = time.time()
            fetch(row.file_id, int(row.file_size), dst)
            t_dl = time.time() - td
            tt = time.time()
            arr, meta = tiles(dst, np.random.RandomState(1000 + k))
            t_tile = time.time() - tt
            te = time.time()
            f = embed(model, arr)
            t_emb = time.time() - te
            feats.append(f)
            index.append(dict(case_id=row.case_id, file_id=row.file_id,
                             file_name=row.file_name, file_size=int(row.file_size),
                             diagnosis=row.diagnosis, t_dl=round(t_dl, 1),
                             t_tile=round(t_tile, 1), t_emb=round(t_emb, 1),
                             **meta, **{g: int(getattr(row, g)) for g in GENES}))
            np.save(os.path.join(outdir, "features.npy"), np.stack(feats))
            pd.DataFrame(index).to_csv(os.path.join(outdir, "index.csv"), index=False)
            print("  [%d/%d] %s  dl=%.0fs tile=%.0fs emb=%.0fs  %s"
                  % (k + 1, len(todo), row.case_id, t_dl, t_tile, t_emb, row.diagnosis), flush=True)
        except Exception as e:
            print("  [%d/%d] %s FAILED %s" % (k + 1, len(todo), row.case_id, str(e)[:110]), flush=True)
            traceback.print_exc()
        finally:
            try:
                if os.path.exists(dst):
                    os.remove(dst)
            except Exception:
                pass
        if (k + 1) % 5 == 0:
            el = time.time() - T0
            print("  == %d/%d done, %.0fs elapsed, %.0fs/slide, ETA %.0f min"
                  % (k + 1, len(todo), el, el / (k + 1), (len(todo) - k - 1) * el / (k + 1) / 60), flush=True)
    print("[saved] %s" % outdir, flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()