"""路径工具函数"""
import sys
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """获取资源文件路径

    开发环境：从项目根目录解析（src/utils/paths.py → src/ → 项目根目录）
    打包环境：使用 PyInstaller/Nuitka 的解包路径
    """
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / relative_path
    elif getattr(sys, "frozen", False):
        return Path(sys.executable).parent / relative_path
    else:
        return Path(__file__).resolve().parent.parent.parent / relative_path
