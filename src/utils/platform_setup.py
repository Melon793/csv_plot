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


def setup_windows_performance() -> list[str]:
    """Windows 专属性能初始化。必须在创建 QApplication 之前调用。

    Returns:
        已应用的措施列表（失败项以 "-failed" 后缀标记，用于启动日志排查）
    """
    applied: list[str] = []
    if sys.platform != "win32":
        return applied

    # 1. 系统定时器分辨率提到 1ms（QTimer 精度从 ~15.6ms 量化提升到 1ms）。
    #    这是 Qt 事件分发器在 Windows 上的已知行为（等待超时按系统时钟
    #    分辨率取整）。
    #    作用域说明（重要）：
    #    - Win11 22H2（2022 起）后，timeBeginPeriod 仅对"调用进程在前台时"
    #      生效，不再全局抬高整个系统的时钟分辨率，也不再显著影响整机功耗；
    #    - Win10 / Win11 21H2 及更早版本仍为全局生效，会禁用部分空闲 C 状态。
    #    因此本调用在所有版本上都是安全的：在新版系统上副作用已被收敛，
    #    在旧版系统上属于"按需付功耗换精度"的合理取舍。
    #    进程退出时建议配对调用 timeEndPeriod(1)（见 cleanup_windows_performance）。
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
        applied.append("timeBeginPeriod(1)")
    except Exception:
        applied.append("timeBeginPeriod-failed")

    return applied


def cleanup_windows_performance() -> None:
    """与 setup_windows_performance 配对的退出清理。

    在 QApplication.aboutToQuit 信号里调用，配对 timeEndPeriod(1)。
    - 长驻 GUI 应用：进程生命周期内不调用也无副作用（进程退出时系统自动回收）；
    - 托盘最小化 / 后台驻留场景：Win11 22H2+ 把 timeBeginPeriod 作用域收到
      "前台进程"，前台态切换会让 1ms 分辨率失效再恢复，行为略复杂——配对
      调用 timeEndPeriod 可让状态切换更干净，作为良好实践推荐实施。
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.winmm.timeEndPeriod(1)
    except Exception:
        pass
