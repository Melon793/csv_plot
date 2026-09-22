"""惰性加载临时目录管理（创建 / 清扫 / atexit 兜底 / 多级回退）。

D17 的三重约束：
1. 基路径固定为 <CacheLocation>/CSVPlot/lazy_tmp —— 刻意加 "CSVPlot"
   这一级，使路径与 Qt 的 app name 解析结果无关（本项目从未调用
   setApplicationName，CacheLocation 会退化成 ~/Library/Caches/<可执行名>，
   开发态与打包态不同，极端情况下落到共享的 Caches 根）。
2. 回退链：writableLocation 返回空串（Qt 在不可写/未知时的行为）或
   mkdir 抛 OSError/PermissionError → tempfile.gettempdir()/CSVPlot_lazy/
   （照 src/core/logger.py 的多级回退写法）。
3. sweep_stale() 三重保险（rmtree 不可逆，宁可漏删不可错删）：
   ① 只扫 base_dir() 这一层，不递归；
   ② 只处理名字匹配 ^\\d+_[0-9a-f]{8,}$ 的条目，其余一律跳过（含文件）；
   ③ 只在「目录名里的 PID 不在进程表」或「目录 mtime 超过 24h」时删。
   第三条防 PID 复用（残留目录的 PID 被别的进程占用 → 永不清理）。

目录名 <pid>_<rand8> 按 loader 实例生成（不是按进程）：同进程先后加载
两个文件时不互相踩。sweep_stale() 绝不删当前存活 PID 的目录（多实例
安全），在应用启动时调用一次。
"""

from __future__ import annotations

import atexit
import os
import re
import secrets
import shutil
import tempfile
import threading
import weakref
from pathlib import Path

from src.core.logger import get_logger

logger = get_logger(__name__)

# 目录名格式：<pid>_<rand8+>；清扫时只认这个格式
_DIR_NAME_RE = re.compile(r"^\d+_[0-9a-f]{8,}$")
# 残留目录 mtime 超过该秒数则视为陈旧（防 PID 复用导致永不清理）
_STALE_AFTER_S = 24 * 3600

# 活动实例登记表：atexit 兜底时对仍然存活的实例做 cleanup。
# WeakSet 不阻止 GC（loader 必须可被回收，variable_info_dialog 的
# weakref 队列依赖这一点）。
_live_dirs: "weakref.WeakSet[TempCacheDir]" = weakref.WeakSet()
_atexit_registered = False
_register_lock = threading.Lock()


def _cache_location() -> str:
    """QStandardPaths.CacheLocation，Qt 不可用时返回空串（走回退链）。"""
    try:
        from PySide6.QtCore import QStandardPaths

        return QStandardPaths.writableLocation(QStandardPaths.CacheLocation) or ""
    except Exception:
        return ""


def _try_mkdir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError):
        return False


class TempCacheDir:
    """一个 loader 实例独占的临时目录（<base>/<pid>_<rand8>/）。"""

    def __init__(self, path: Path):
        self._path = path
        self._cleaned = False

    # ---- 路径 ----
    @staticmethod
    def base_dir() -> Path:
        """惰性数据临时目录的基路径（带多级回退，保证可写）。"""
        cache_loc = _cache_location()
        if cache_loc:
            primary = Path(cache_loc) / "CSVPlot" / "lazy_tmp"
            if _try_mkdir(primary):
                return primary
        fallback = Path(tempfile.gettempdir()) / "CSVPlot_lazy"
        # 回退路径理论一定可写（gettempdir 的契约）；再失败只能抛
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

    @staticmethod
    def create() -> "TempCacheDir":
        """创建一个新目录并登记到 atexit 兜底表。"""
        path = TempCacheDir.base_dir() / f"{os.getpid()}_{secrets.token_hex(4)}"
        path.mkdir(parents=True, exist_ok=False)
        inst = TempCacheDir(path)
        with _register_lock:
            _live_dirs.add(inst)
        return inst

    def path(self) -> str:
        return str(self._path)

    # ---- 清理 ----
    def cleanup(self) -> None:
        """删除本目录（幂等；loader.close() 与 atexit 兜底都走这里）。"""
        if self._cleaned:
            return
        self._cleaned = True
        shutil.rmtree(self._path, ignore_errors=True)

    # ---- 启动清扫 ----
    @staticmethod
    def sweep_stale(base: Path | None = None) -> int:
        """清扫残留目录（应用启动时调用一次），返回删除数。

        三重保险见模块 docstring；任何一条不满足都跳过该条目。
        """
        base = base if base is not None else TempCacheDir.base_dir()
        own_pid = os.getpid()
        removed = 0
        try:
            entries = list(base.iterdir())
        except OSError:
            return 0
        for entry in entries:
            # 保险②：名字不匹配的一律跳过（含文件，含其它命名的目录）
            if not _DIR_NAME_RE.match(entry.name):
                continue
            if not entry.is_dir():
                continue
            # 保险③-a：PID 仍存活且目录未超时 → 保留（多实例安全）
            pid = int(entry.name.split("_", 1)[0])
            if pid == own_pid or _pid_alive(pid):
                if _mtime_age_s(entry) <= _STALE_AFTER_S:
                    continue
            # 保险③-b：PID 已死（主判据）或 mtime 超 24h（防 PID 复用）→ 删
            # 保险①：只删本层匹配目录本身（rmtree 会带走其内容，但入口
            # 只来自 base_dir 这一层 iterdir，不会递归进入别人的目录结构）
            shutil.rmtree(entry, ignore_errors=True)
            removed += 1
        return removed

    # ---- atexit 兜底 ----
    @staticmethod
    def register_atexit() -> None:
        """进程级兜底：正常退出时清掉所有活动实例的目录（幂等注册）。"""
        global _atexit_registered
        with _register_lock:
            if _atexit_registered:
                return
            _atexit_registered = True
            atexit.register(TempCacheDir._atexit_cleanup)

    @staticmethod
    def _atexit_cleanup() -> None:
        for inst in list(_live_dirs):
            try:
                inst.cleanup()
            except Exception:  # noqa: BLE001 - 退出路径绝不抛
                pass


def _pid_alive(pid: int) -> bool:
    """PID 是否在进程表（os.kill 信号 0 探测；EPERM 视为存活）。"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 进程存在但归别的用户所有
    except OSError:
        return False


def _mtime_age_s(entry: Path) -> float:
    try:
        import time

        return max(0.0, time.time() - entry.stat().st_mtime)
    except OSError:
        return 0.0
