"""
test_sdpc.py — WSL Ubuntu 里验证 opensdpc 能否读 .sdpc

用法(WSL Ubuntu 终端):
    /home/admin123/.venv-pancaner/bin/python /mnt/d/pan-caner/tools/test_sdpc.py

脚本内部会自动把 opensdpc 自带的 .so 加进 LD_LIBRARY_PATH 并自重启,
所以不需要 source venv,也不需要手动 export。

期望输出(8 行):
    [OK] opened 895983003.sdpc
    level_count: 4
    magnification: 40
    sampling_rate: 0.5
    level_dimensions: ((6912, 6912), (3456, 3456), (1728, 1728), (864, 864))
    thumbnail size: (864, 864) mode: RGB
    [OK] saved /tmp/sdpc_thumb_test.jpg
"""
# IMPORTANT: set LD_LIBRARY_PATH for opensdpc native libs BEFORE importing
# opensdpc. ctypes.CDLL triggers on import, and on Linux it needs
# libDecodeSdpc.so / libDecodeHevc.so / bundled ffmpeg & jpeg (the .so files
# ship inside the opensdpc pip wheel but are not on the default loader path).
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _opensdpc_runtime import setup_opensdpc_libpath  # noqa: E402

setup_opensdpc_libpath()

import opensdpc  # noqa: E402  (must come after the LD_LIBRARY_PATH setup)
from PIL import Image

SDPC_PATH = "/mnt/d/pan-caner/data/中日冰冻切片/895983003.sdpc"
THUMB_PATH = "/tmp/sdpc_thumb_test.jpg"

slide = opensdpc.OpenSdpc(SDPC_PATH)
print(f"[OK] opened {SDPC_PATH.rsplit('/', 1)[-1]}")
print("level_count:", slide.level_count)
print("magnification:", slide.scan_magnification)
print("sampling_rate:", slide.sampling_rate)
print("level_dimensions:", slide.level_dimensions)

thumb = slide.get_thumbnail(slide.level_count - 1)
if not isinstance(thumb, Image.Image):
    thumb = Image.fromarray(thumb)
print("thumbnail size:", thumb.size, "mode:", thumb.mode)
thumb.save(THUMB_PATH, "JPEG")
print(f"[OK] saved {THUMB_PATH}")

# 顺便试 patch 读取(从 level 0 读一个 1024x1024 patch)
try:
    patch = slide.read_region((0, 0), 0, (1024, 1024))
    if not isinstance(patch, Image.Image):
        patch = Image.fromarray(patch)
    patch.save("/tmp/sdpc_patch_test.jpg", "JPEG")
    print(f"[OK] saved /tmp/sdpc_patch_test.jpg (1024x1024 patch from level 0)")
except Exception as e:
    print(f"[WARN] patch read failed: {e}")

print("DONE")
