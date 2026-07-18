"""
_opensdpc_runtime.py — 让 opensdpc 在 Linux/WSL 下无需手动 export 就能加载 .so

直接 import 这个模块,任何在它之后 import opensdpc 的代码就都能跑了。
- 自动定位 venv 内 opensdpc 包自带的 LINUX/(ffmpeg|jpeg) 子目录
- 加进 LD_LIBRARY_PATH
- os.execv 自重启让动态链接器在子进程上认到新路径
  (ctypes.CDLL 是进程级一次性解析,中途改 env 没用)
"""
import os
import sys
from pathlib import Path

_MARKER = "_SDPC_LD_REEXEC"


def _opensdpc_linux_dir() -> Path | None:
    """根据当前 Python 找 opensdpc 包里的 LINUX/ 子目录"""
    # sys.prefix 指向 venv 根或系统 Python 根
    candidates = [
        Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "opensdpc" / "LINUX",
        Path(sys.prefix) / "lib" / "site-packages" / "opensdpc" / "LINUX",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def setup_opensdpc_libpath() -> None:
    """把 opensdpc 自带的 .so 加进 LD_LIBRARY_PATH,必要时 re-exec 自己"""
    linux_dir = _opensdpc_linux_dir()
    if linux_dir is None:
        return  # 找不到就不做,让后续 import 自然失败给明确错误

    parts = [str(linux_dir), str(linux_dir / "ffmpeg"), str(linux_dir / "jpeg")]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if existing:
        parts.append(existing)
    os.environ["LD_LIBRARY_PATH"] = ":".join(parts)

    if os.environ.get(_MARKER) != "1":
        os.environ[_MARKER] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
