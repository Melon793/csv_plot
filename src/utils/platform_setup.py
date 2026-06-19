"""平台初始化模块 — 集中处理跨平台环境变量和图标设置"""

from __future__ import annotations
import sys
import os


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
