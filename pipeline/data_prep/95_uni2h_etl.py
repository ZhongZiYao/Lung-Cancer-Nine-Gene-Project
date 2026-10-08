"""95_uni2h_etl.py - 解压 MahmoodLab/UNI2-h-features 的 CPTAC tar 并探查 h5 结构"""
import os, sys, time, tarfile, h5py, glob, shutil
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 源 tar 实际在嵌套一层 (94 脚本的 dst 拼接 bug 制造的 CPTAC/CPTAC/)
SRC_DIR = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\public_features\MahmoodLab_UNI2-h\CPTAC\CPTAC"
DST = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\public_features\MahmoodLab_UNI2-h\CPTAC_extracted"
os.makedirs(DST, exist_ok=True)
TARS = [os.path.join(SRC_DIR, "cptac_luad_part1.tar.gz"),
        os.path.join(SRC_DIR, "cptac_luad_part2.tar.gz")]
print("DST =", DST, "  disk free =", round(shutil.disk_usage(DST).free / 1e9, 1), "GB", flush=True)

for tar_path in TARS:
    if not os.path.exists(tar_path):
        print("MISSING:", tar_path, flush=True); continue
    print("\n=== extracting", os.path.basename(tar_path), flush=True)
    t0 = time.time()
    cnt = 0
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf:
            if not member.isfile(): continue
            tgt = os.path.join(DST, member.name.replace("/", os.sep))
            os.makedirs(os.path.dirname(tgt), exist_ok=True)
            if os.path.exists(tgt): cnt += 1; continue
            with tf.extractfile(member) as src, open(tgt, "wb") as out:
                while True:
                    b = src.read(1 << 24)
                    if not b: break
                    out.write(b)
            cnt += 1
            if cnt % 50 == 0:
                print(f"  {cnt} files  {os.path.getsize(tgt)/1e6:.1f}MB last  [{time.time()-t0:.0f}s]", flush=True)
    sz = sum(os.path.getsize(os.path.join(r,f)) for r,_,fs in os.walk(DST) for f in fs)
    print(f"=== done {os.path.basename(tar_path)}  files={cnt}  size={sz/1e9:.2f}GB  time={(time.time()-t0)/60:.1f}min", flush=True)

print("\n=== 探查 h5 结构 ===", flush=True)
h5s = sorted(glob.glob(os.path.join(DST, "**", "*.h5"), recursive=True))[:5]
print("示例 h5 文件 (前 5):", flush=True)
for h in h5s:
    print("  ", os.path.basename(h), flush=True)
if h5s:
    for h in h5s:
        with h5py.File(h, "r") as f:
            print(f"\n--- {os.path.basename(h)} ---", flush=True)
            for k in list(f.keys()):
                obj = f[k]
                if isinstance(obj, h5py.Dataset):
                    print(f"  Dataset {k!r}  shape={obj.shape}  dtype={obj.dtype}", flush=True)
                else:
                    print(f"  Group {k!r}  keys={list(obj.keys())[:8]}", flush=True)
                    for sk in list(obj.keys())[:4]:
                        sobj = obj[sk]
                        if isinstance(sobj, h5py.Dataset):
                            print(f"    {sk!r}  shape={sobj.shape}  dtype={sobj.dtype}", flush=True)
                        else:
                            print(f"    {sk!r}  type=Group  keys={list(sobj.keys())[:5]}", flush=True)

print("\n=== case_id 采样 ===", flush=True)
ids = set()
for h in h5s[:50] + sorted(glob.glob(os.path.join(DST, "**", "*.pt"), recursive=True))[:50]:
    base = os.path.splitext(os.path.basename(h))[0]
    head = base.split("-")[0] if "-" in base else base
    ids.add(head)
print("示例 case id (前 15):", sorted(ids)[:15], flush=True)
print("唯一 case 数:", len(ids), flush=True)
print("DONE", flush=True)