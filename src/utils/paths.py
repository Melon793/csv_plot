"""路径工具函数"""
from pathlib import Path
from src.utils.platform_setup import get_bundle_dir


def resource_path(relative_path: str) -> Path:
    """获取资源文件路径

    开发环境：从项目根目录解析
    打包环境：使用统一的 bundle_dir 定位
    """
    return get_bundle_dir() / relative_path
