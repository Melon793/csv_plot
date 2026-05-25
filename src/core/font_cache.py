"""
Font cache layer —— 基于 QSettings 的字体缓存, 避免每次启动枚举系统字体

在 Windows 上, QFontDatabase.families() 会枚举所有已安装字体 (~50-100ms),
缓存后仅首次/字体变更时触发枚举。

并发安全说明:
  QSettings 是 reentrant 的, 在 QApplication 创建后的主线程中调用是安全的。
  字体必须在 MainWindow 构造之前确定, 因此缓存必须是同步操作,
  不能使用异步方案。本模块在 QApplication 创建之后、MainWindow 构造之前调用。

缓存策略:
  - 写入: 首次检测成功后写入 QSettings
  - 读取: 后续启动直接从 QSettings 读取, 跳过 families() 枚举
  - 失效: 当缓存字体名不在当前系统中时回退到重新枚举
  - 版本: CACHE_VERSION 递增可强制全局重新枚举
"""

from __future__ import annotations

CACHE_VERSION = 1

_FONT_PRIORITY_WIN = [
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS",
]


def _detect_font_win() -> str:
    """枚举系统字体并返回第一个匹配的中文字体名"""
    from PySide6.QtGui import QFontDatabase

    available = QFontDatabase.families()
    for name in _FONT_PRIORITY_WIN:
        if name in available:
            return name
    return ""


_MONO_FONT_PRIORITY = [
    "SF Mono",
    "JetBrains Mono",
    "Fira Code",
    "Cascadia Code",
    "Menlo",
    "Consolas",
    "Monaco",
    "DejaVu Sans Mono",
    "Noto Sans Mono",
    "Source Code Pro",
    "Courier New",
]

_MONO_CACHE_VERSION = 1


def _detect_mono_font() -> str:
    """枚举系统字体并返回第一个匹配的等宽字体名"""
    from PySide6.QtGui import QFontDatabase

    available = QFontDatabase.families()
    for name in _MONO_FONT_PRIORITY:
        if name in available:
            return name
    return ""


def get_monospace_font_cached() -> str:
    """
    返回缓存或检测到的等宽字体名。

    搭配 QFont(name, pixel_size) 使用。
    返回空字符串则回退到 QFont("monospace") 默认行为。
    """
    from PySide6.QtGui import QFontDatabase
    from src.core.settings import AppSettings

    settings = AppSettings()
    cached_version = settings.get_mono_font_cache_version()
    if cached_version == _MONO_CACHE_VERSION:
        cached_name = settings.get_mono_font_name()
        if cached_name and cached_name in QFontDatabase.families():
            return cached_name

    detected = _detect_mono_font()
    if not detected:
        return ""

    settings.set_mono_font_cache_version(_MONO_CACHE_VERSION)
    settings.set_mono_font_name(detected)
    return detected


def get_windows_chinese_font_cached() -> str:
    """
    返回缓存或检测到的中文字体名。

    写缓存不显式返回给调用方, 仅作为副作用保存字体名。
    下游用返回的名字创建 QFont(name, pixel_size)。

    返回空字符串时调用方应回退到 QApplication.font()。
    """
    from PySide6.QtGui import QFontDatabase
    from src.core.settings import AppSettings

    settings = AppSettings()
    cached_version = settings.get_font_cache_version()
    if cached_version == CACHE_VERSION:
        cached_name = settings.get_font_name()
        if cached_name and cached_name in QFontDatabase.families():
            return cached_name

    detected = _detect_font_win()
    if not detected:
        return ""

    settings.set_font_cache_version(CACHE_VERSION)
    settings.set_font_name(detected)
    return detected
