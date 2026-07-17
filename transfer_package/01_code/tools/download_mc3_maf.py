"""
download_mc3_maf.py — 拉 MC3 pan-cancer 公开 MAF (TCGA-LUAD 9 基因标签来源)

数据源:
  Synapse syn7824274 = mc3.v0.2.8.PUBLIC.maf.gz
  链接: https://www.synapse.org/Synapse:syn7824274
  大小: 几百 MB (压缩) / ~2 GB (解压 ~10M 行 MAF)

需要:
  - 注册 Synapse 账号 (1 min) https://www.synapse.org/SignUp
  - 拿 Personal Access Token: 用户名 → Account Settings → Personal Access Tokens
  - 把 token 写到 .synapse_token 文件 (默认 D:/pan-caner/.synapse_token),或环境变量 SYNAPSE_TOKEN

用法:
  /home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/download_mc3_maf.py
  /home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/download_mc3_maf.py --out /mnt/d/pan-caner/data/MC3

说明:
  - 唯一支持路径: Synapse + PAT (syn7824274 是 PUBLIC,无需 dbGaP,只需 Synapse 账号)
  - 旧版曾试 HTTPS 抓 https://www.synapse.org/Synapse:syn7824274,只能拿到 HTML,已删除
"""
import argparse
import os
import sys
from pathlib import Path

MC3_SYNAPSE_ID = "syn7824274"
MC3_FILENAME = "mc3.v0.2.8.PUBLIC.maf.gz"


def get_token(token_arg: str | None) -> str:
    """从 --token / .synapse_token / SYNAPSE_TOKEN 环境变量拿 token"""
    if token_arg:
        return token_arg
    env = os.environ.get("SYNAPSE_TOKEN")
    if env:
        return env
    candidates = [
        Path.cwd() / ".synapse_token",
        Path.home() / ".synapse_token",
        Path("/mnt/d/pan-caner/.synapse_token"),
    ]
    for p in candidates:
        if p.is_file():
            return p.read_text().strip()
    raise RuntimeError(
        "No Synapse token. Either:\n"
        "  1) 注册 https://www.synapse.org/SignUp →  Account Settings → Personal Access Tokens\n"
        f"  2) 把 token 写到 {candidates[2]}\n"
        "     或环境变量 export SYNAPSE_TOKEN=...\n"
        "  3) 或重跑时加 --token <your-pat>"
    )


def download_via_synapse(out_dir: Path) -> Path:
    import synapseclient

    syn = synapseclient.Synapse()
    syn.login(authToken=os.environ["_SYNAPSE_AUTH"])
    print(f"[synapse] logged in as {syn.getUserProfile()['userName']}")

    ent = syn.get(entity=MC3_SYNAPSE_ID, downloadLocation=str(out_dir))
    target = Path(ent.path) if ent.path else out_dir / MC3_FILENAME
    if not target.exists():
        raise RuntimeError(f"synapseclient didn't download file to {target}")
    return target


def main() -> int:
    ap = argparse.ArgumentParser(description="Download MC3 pan-cancer public MAF for 9-gene label extraction")
    ap.add_argument("--out", default="/mnt/d/pan-caner/data/MC3", help="output directory")
    ap.add_argument("--token", help="Synapse PAT (else read .synapse_token / $SYNAPSE_TOKEN)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = get_token(args.token)
    os.environ["_SYNAPSE_AUTH"] = token
    target = download_via_synapse(out_dir)

    size_mb = target.stat().st_size / 1e6
    print(f"\n[done] {target}  ({size_mb:.1f} MB)")
    print(f"       next: python tools/extract_9gene_panel.py --maf {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
