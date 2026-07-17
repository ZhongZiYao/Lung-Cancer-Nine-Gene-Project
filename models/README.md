# Third-party Model Sources

This directory contains **third-party** model repositories we vendored as reference code (no weights).

We do **not** maintain these subdirectories as git submodules. We include the source code only so you can `cd` into them and read the architecture. For training/inference, you must clone the originals separately and put weights in the right places.

| Path | Repo URL | What we use |
|---|---|---|
| `../CONCH/` | https://github.com/mahmoodlab/CONCH.git | vision-language encoder, backup patch encoder |
| `../DeepGEM/` | https://github.com/TencentAILabHealthcare/DeepGEM.git | baseline architecture + their pretrained 9-gene models |
| `../TITAN/` | https://github.com/mahmoodlab/TITAN.git | spatial aggregator reference |
| `../TransPath/` | https://github.com/Xiyue-Wang/TransPath.git | CTransPath loader source (we copy `net/models/modeling.py`) |
| `../UNI/` | https://github.com/mahmoodlab/UNI.git | **primary** patch encoder (1024-dim CLS) |
| `../Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision/` | https://github.com/zjsmzn/Prediction-of-Mutated-Genes-in-Lung-Adenocarcinoma-Based-on-Weak-Supervision.git | upstream 9-gene project (THUML GAMIL, reference) |

## Vendored without weights

| Vendor | Size w/o weights | License | What we exclude |
|---|---|---|---|
| CONCH | 1.5 GB | CC-BY-NC | `checkpoints/CONCH/pytorch_model.bin` (model weights) |
| DeepGEM | 12 GB | MIT | `checkpoints/DeepGEM_TCGA/*.pickle` (pretrained gene models), `timm-0.5.4.tar` |
| TITAN | 12 GB | CC-BY-NC | `output.csv` (12 GB!), `datasets/TCGA-Slide-Reports.csv.gz` |
| TransPath | < 10 MB | (see repo) | — |
| UNI | 2 GB | CC-BY-NC | `assets/ckpts/uni/pytorch_model.bin` (1.2 GB) |
| Prediction-of-Mutated-Genes... | < 50 MB | MIT | — |

## Where to get weights

```bash
# UNI
cd UNI
git lfs install
git lfs pull  # if .gitattributes says lfs
# OR:
wget -O assets/ckpts/uni/pytorch_model.bin \
  https://huggingface.co/MahmoodLab/UNI/resolve/main/pytorch_model.bin

# CTransPath (in DeepGEM)
cd DeepGEM/checkpoints/pretrain
wget -O ctranspath.pth https://download.pytorch.org/...ctranspath.pth
# (or use TransPath project README)

# DeepGEM 训好的 9-gene models
# 9 个 .pickle 文件，DeepGEM 文档里给 huggingface 链接
```

## Why not git submodule

Submodules add clone complexity (`git submodule update --init`). And the original `uni/`, `deepgem/` etc are 1-12 GB each, unfit for a clean "code + data + thesis" repo. This is a **research project** repo, not a deployment artifact, so vendored code is OK.

If you DO want proper submodules later, here's the recipe:

```bash
# Re-init each as a submodule (keep the data separate)
cd lung-cancer-nine-gene-project
git submodule add https://github.com/mahmoodlab/UNI.git UNI
git submodule add https://github.com/TencentAILabHealthcare/DeepGEM.git DeepGEM
# ... etc for the other 4
git commit -m "Re-add third-party repos as git submodules"
```

## License

Each third-party repo retains its original license. See each repo's LICENSE file.
