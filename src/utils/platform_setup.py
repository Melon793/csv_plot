"""平台初始化模块 — 集中处理跨平台环境变量、打包检测和图标设置"""

from __future__ import annotations
import sys
import os
from pathlib import Path


def is_frozen() -> bool:
    """检测是否处于打包环境（PyInstaller / Nuitka / cx_Freeze）。

    Returns:
        True 如果应用被打包为独立可执行文件
    """
    # PyInstaller
    if getattr(sys, "frozen", False):
        return True
    # Nuitka
    import __main__
    if "__compiled__" in getattr(__main__, "__dict__", {}):
        return True
    return False


def get_bundle_dir() -> Path:
    """获取打包环境下的资源目录。

    Returns:
        - PyInstaller: sys._MEIPASS (临时解包目录)
        - Nuitka/cx_Freeze: exe 所在目录
        - 开发环境: 项目根目录

    Raises:
        RuntimeError: 打包环境下无法确定目录时抛出
    """
    if hasattr(sys, "_MEIPASS"):
        # PyInstaller
        return Path(sys._MEIPASS)

    if getattr(sys, "frozen", False):
        # cx_Freeze / 其他 frozen 工具
        exe_dir = Path(sys.executable).parent
        if exe_dir.exists():
            return exe_dir

    # Nuitka (__compiled__ 但非 frozen)
    import __main__
    if "__compiled__" in getattr(__main__, "__dict__", {}):
        exe_dir = Path(sys.executable).parent
        if exe_dir.exists():
            return exe_dir

    # 开发环境
    return Path(__file__).resolve().parents[2]


def setup_platform() -> str | None:
    """设置平台环境变量并返回图标路径。

    - 设置 PYQTGRAPH_QT_LIB 为 PySide6
    - Windows: 设置 AppUserModelID 并返回 .ico 路径
    - macOS: 返回 .icns 路径
    - 其他平台: 返回 None

    Returns:
        图标文件路径，如果不适用于当前平台则返回 None
    """
    os.environ["PYQTGRAPH_QT_LIB"] = "PySide6"

    from src.utils.paths import resource_path

    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "mycompany.csv_plot.0.1"
        )
        return resource_path("assets/icon.ico")
    elif sys.platform == "darwin":
        return resource_path("assets/icon.icns")
    return None
