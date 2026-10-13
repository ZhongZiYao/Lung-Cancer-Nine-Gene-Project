import sys, os, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
C = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache"
for f in ["NSCLC_gigap_sub512_index.csv", "NSCLC_gigap_sub4096_index.csv",
          "NSCLC_gigap_luad_sub2048_index.csv", "NSCLC_uni_sub512_index.csv",
          "NSCLC_chief_sub512_index.csv", "NSCLC_uni2h_luad_sub512_index.csv",
          "CPTAC_uni_sub512_index.csv"]:
    p = os.path.join(C, f)
    if not os.path.exists(p): continue
    d = pd.read_csv(p)
    if "n_patch_orig" in d.columns:
        o = d.n_patch_orig
        u = d.get("n_patch_used", pd.Series([np.nan]*len(d)))
        print("%-34s n=%-5d  n_patch_orig 中位=%-6.0f  min=%-6.0f max=%-7.0f  用了=%s"
              % (f, len(d), o.median(), o.min(), o.max(),
                 (u.median() if u.notna().any() else "-")), flush=True)
    else:
        print("%-34s n=%-5d  列=%s" % (f, len(d), [x for x in d.columns][:5]), flush=True)