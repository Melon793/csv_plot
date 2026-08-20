"""utils.paths 单元测试：资源路径解析。"""

from __future__ import annotations

from pathlib import Path

from src.utils.paths import resource_path


def test_resource_path_joins_relative():
    p = resource_path("assets/icon.png")
    assert isinstance(p, Path)
    assert p.as_posix().endswith("assets/icon.png")


def test_existing_asset_resolves():
    """开发环境下项目自带的 assets 应可解析到存在的文件"""
    p = resource_path("assets/icon.png")
    assert p.exists()
