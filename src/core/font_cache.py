"""
Font cache layer —— 基于 QSettings 的字体缓存, 避免每次启动枚举系统字体

在 Windows 上, QFontDatabase.families() 会枚举所有已安装字体 (~50-100ms),
缓存后仅首次/字体变更时触发枚举。

Windows 界面字体用**字体栈**而不是单一中文字体：拉丁走系统 UI 字体、
汉字落到中文字体。理由是英文 Windows 上整屏拉丁文本被钉成中文 UI 字体
（雅黑拉丁字形/行高）与原生应用不一致；而缺字时单族只能交给平台隐式
回退，落到哪套字形取决于该机已装的 CJK 字体与系统语言。

并发安全说明:
  QSettings 是 reentrant 的, 在 QApplication 创建后的主线程中调用是安全的。
  字体必须在 MainWindow 构造之前确定, 因此缓存必须是同步操作,
  不能使用异步方案。本模块在 QApplication 创建之后、MainWindow 构造之前调用。

缓存策略:
  - 写入: 首次检测成功后写入 QSettings
  - 读取: 后续启动直接从 QSettings 读取, 只做一次字形覆盖校验
  - 失效: 缓存字体名不存在或画不出常用汉字时回退到重新检测
  - 版本: CACHE_VERSION 递增可强制全局重新检测
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PySide6.QtGui import QFont

CACHE_VERSION = 2  # 2：校验从「字体名存在」改为「真的覆盖常用汉字」

#: Windows 拉丁文本首选：Win11 的系统 UI 字体，Win10 上回落到 Segoe UI。
_UI_FONT_STACK_WIN = [
    "Segoe UI Variable Text",
    "Segoe UI",
]

#: Windows 中文字体优先级。只列真正可能覆盖汉字的族：
#: SimHei 属「中文补充字体」可选功能包（FOD），Arial Unicode MS 不在
#: Windows 自带清单（随 Office 安装），两者都不能当地基。
_FONT_PRIORITY_WIN = [
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "DengXian",
    "SimHei",
]

#: 覆盖校验探针：界面上一定会出现的常用汉字
_CHINESE_PROBE = "中曲线"

#: 白名单全落空时给系统 zh 字体池排序用（仅比较小写名）
_SANS_HINTS = (
    "yahei",
    "dengxian",
    "simhei",
    "noto sans",
    "source han",
    "sourcehan",
    "misans",
    "harmonyos",
    "pingfang",
)
_SERIF_HINTS = (
    "simsun",
    "nsimsun",
    "fangsong",
    "kaiti",
    "mingliu",
    "dfkai",
    "song",
    "楷",
    "宋",
    "仿宋",
)

_installed_families: frozenset[str] | None = None
_resolved_chinese_family: str = ""


def installed_font_families() -> frozenset[str]:
    """进程级「已安装字体族」快照，问一次记下来。

    `QFontDatabase.families()` 在 Windows 上要枚举全部字体（~50-100 ms），而它过去
    被放在缓存命中的校验里（等于把缓存要省的时间原样花回去）以及 splash 的绘制
    路径上（每帧一次）。一次运行里字体表不会变，成员判断用它即可。
    """
    global _installed_families
    if _installed_families is None:
        from PySide6.QtGui import QFontDatabase

        _installed_families = frozenset(QFontDatabase.families())
    return _installed_families


def _detect_font(priority_list: list[str]) -> str:
    """按优先级列表检测第一个可用的字体名。

    Args:
        priority_list: 按优先级排序的字体名列表

    Returns:
        第一个可用的字体名，或空字符串
    """
    available = installed_font_families()
    for name in priority_list:
        if name in available:
            return name
    return ""


def _family_supports_chinese(family: str) -> bool:
    """该字体族是否真的画得出探针汉字。

    「名字出现在 families() 里」不等于「能画中文」：字体缺失或被替换时，
    setFont 会静默落到平台隐式回退——英文系统上可能是日文/繁中/宋体字形，
    同一条文本里还会混排两种字体。这里用 QRawFont 直接问字形覆盖。
    """
    from PySide6.QtGui import QFont, QRawFont

    probe = QFont(family)
    probe.setPixelSize(12)
    raw = QRawFont.fromFont(probe)
    return raw.isValid() and all(raw.supportsCharacter(ord(ch)) for ch in _CHINESE_PROBE)


def _chinese_families_from_system() -> list[str]:
    """系统声明支持简体中文的字体族，无衬线黑体优先、衬线/书法体最后。

    只含第三方中文字体（思源、MiSans…）或精简镜像时才走这条：至少别让
    界面落到宋体/楷体上。
    """
    from PySide6.QtGui import QFontDatabase

    families = QFontDatabase.families(QFontDatabase.WritingSystem.SimplifiedChinese)

    def rank(name: str) -> tuple[int, str]:
        lowered = name.casefold()
        if any(hint in lowered for hint in _SANS_HINTS):
            return (0, lowered)
        if any(hint in lowered for hint in _SERIF_HINTS):
            return (2, lowered)
        return (1, lowered)

    return sorted(families, key=rank)


def _detect_font_win() -> str:
    """按优先级返回第一个真正覆盖常用汉字的字体族，全无则返回空串。"""
    for name in _FONT_PRIORITY_WIN:
        if _family_supports_chinese(name):
            return name
    for name in _chinese_families_from_system():
        if _family_supports_chinese(name):
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
    return _detect_font(_MONO_FONT_PRIORITY)


def get_monospace_font_cached() -> str:
    """
    返回缓存或检测到的等宽字体名。

    搭配 QFont(name, pixel_size) 使用。
    返回空字符串则回退到 QFont("monospace") 默认行为。
    """
    from src.core.settings import AppSettings

    settings = AppSettings()
    cached_version = settings.get_mono_font_cache_version()
    if cached_version == _MONO_CACHE_VERSION:
        cached_name = settings.get_mono_font_name()
        if cached_name and cached_name in installed_font_families():
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

    命中缓存时只做一次字形覆盖校验；校验不过（字体被卸载/被替换/缓存来自
    另一台机器）就重新检测。返回空字符串表示本机没有能画常用汉字的中文字体。
    """
    global _resolved_chinese_family

    from src.core.settings import AppSettings

    settings = AppSettings()
    cached_version = settings.get_font_cache_version()
    if cached_version == CACHE_VERSION:
        cached_name = settings.get_font_name()
        if cached_name and _family_supports_chinese(cached_name):
            _resolved_chinese_family = cached_name
            return cached_name

    detected = _detect_font_win()
    _resolved_chinese_family = detected
    if not detected:
        return ""

    settings.set_font_cache_version(CACHE_VERSION)
    settings.set_font_name(detected)
    return detected


def _windows_ui_font_families(cjk_family: str) -> list[str]:
    """字体栈：系统拉丁字体在前，中文字体随后，其余候选兜底且去重。"""
    families: list[str] = []
    for name in (*_UI_FONT_STACK_WIN, cjk_family, *_FONT_PRIORITY_WIN):
        if name and name not in families:
            families.append(name)
    return families


def build_windows_ui_font(pixel_size: int = 12) -> "QFont":
    """Windows 应用字体：字体栈 + 像素字号。

    选中的中文字体同时记进 ``_resolved_chinese_family``，供轴标签这类
    用 HTML/QSS 单独指定 font-family、拿不到字体栈的路径复用。
    启动日志记录请求栈/实际解析族，现场字体异常可直接从日志定位。
    """
    from PySide6.QtGui import QFont, QFontInfo

    from src.core.logger import get_logger

    cjk_family = get_windows_chinese_font_cached()
    families = _windows_ui_font_families(cjk_family)

    font = QFont()
    font.setFamilies(families)
    font.setPixelSize(pixel_size)

    logger = get_logger("app.font")
    logger.info(
        "Windows 界面字体栈=%s；中文字体=%s；实际解析=%s；字号=%dpx",
        families,
        cjk_family or "<缺失>",
        QFontInfo(font).family(),
        pixel_size,
    )
    if not cjk_family:
        logger.warning("未找到覆盖常用汉字的中文字体，中文可能显示为方块或系统临时回退字体")
    return font


def resolved_chinese_family() -> str:
    """本次运行解析到的中文字体族（未在 Windows 上检测时为空串）。"""
    return _resolved_chinese_family