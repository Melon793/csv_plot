"""版本与构建信息查询。

优先级：_build_info.py (构建产物) > pyproject.toml (开发环境) > importlib.metadata (安装环境) > "unknown"
"""
from __future__ import annotations

import pathlib


def get_version() -> str:
    """获取应用版本号。"""
    try:
        from src._build_info import __version__
        return __version__
    except ImportError:
        pass
    # 开发环境：项目未安装为发行版（pyproject.toml 无 [build-system]），
    # importlib.metadata 查不到，直接解析源码树中的 pyproject.toml
    try:
        import tomllib

        pyproject = pathlib.Path(__file__).resolve().parent.parent / "pyproject.toml"
        with pyproject.open("rb") as f:
            return tomllib.load(f)["project"]["version"]
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("csv-plot")
    except Exception:
        return "unknown"


def get_build_time() -> str:
    """获取编译时间。仅构建后产物有值，开发环境返回空字符串。"""
    try:
        from src._build_info import __build_time__
        return __build_time__
    except ImportError:
        return ""
