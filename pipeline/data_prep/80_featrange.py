import numpy as np, os
C = r"E:\Program Files\pan-cancer\nine-gene-model-for-lung-cancer\data\patch_cache"
for f in ["NSCLC_gigap_sub512.npy", "NSCLC_uni_sub512.npy", "NSCLC_uni2h_luad_sub512.npy"]:
    X = np.load(os.path.join(C, f), mmap_mode="r")
    a = np.asarray(X[:3], dtype=np.float32)
    z = (a / 255.0 - 0.5) / 0.5
    print("%-32s shape=%s" % (f, X.shape), flush=True)
    print("    raw   : min=%.4f max=%.4f mean=%.4f std=%.4f" % (a.min(), a.max(), a.mean(), a.std()), flush=True)
    print("    after pixel-norm (x/255-0.5)/0.5 : mean=%.6f std=%.6f" % (z.mean(), z.std()), flush=True)