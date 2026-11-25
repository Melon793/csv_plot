"""Windows 字体栈：拉丁走系统 UI 字体、汉字落到真正覆盖汉字的中文字体。

历史实现是「白名单取第一个存在的名字 + 单一 family」。两条缺陷：
1. 存在 ≠ 能画中文：字体被替换/无字形时 setFont 静默落到平台隐式回退
   （英文 Windows 上可能是日文/繁中/宋体字形）；
2. 单族意味着整屏拉丁文本也被钉成中文 UI 字体，与原生应用观感不一致。
这里钉死字体栈顺序、字形覆盖校验与缓存自愈三条契约。
"""

from __future__ import annotations

import pytest

from PySide6.QtGui import QFontDatabase

from src.core import font_cache as fc


class _FontSettingsStub:
    def __init__(self, name: str = "", version: int = 0):
        self.name = name
        self.version = version

    def get_font_cache_version(self) -> int:
        return self.version

    def get_font_name(self) -> str:
        return self.name

    def set_font_cache_version(self, version: int) -> None:
        self.version = version

    def set_font_name(self, name: str) -> None:
        self.name = name


class _DummyLogger:
    def __init__(self):
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, message, *args):
        self.infos.append(message % args)

    def warning(self, message, *args):
        self.warnings.append(message % args)


@pytest.fixture(autouse=True)
def reset_resolved_family(monkeypatch):
    monkeypatch.setattr(fc, "_resolved_chinese_family", "")


class TestFontStack:
    def test_latin_first_then_detected_chinese(self):
        assert fc._windows_ui_font_families("Microsoft YaHei UI") == [
            "Segoe UI Variable Text",
            "Segoe UI",
            "Microsoft YaHei UI",
            "Microsoft YaHei",
            "DengXian",
            "SimHei",
        ]

    def test_missing_chinese_family_leaves_no_empty_entry(self):
        families = fc._windows_ui_font_families("")

        assert families[:2] == ["Segoe UI Variable Text", "Segoe UI"]
        assert "" not in families
        assert "Microsoft YaHei UI" in families

    def test_third_party_chinese_font_follows_latin(self):
        families = fc._windows_ui_font_families("Noto Sans SC")

        assert families[:3] == ["Segoe UI Variable Text", "Segoe UI", "Noto Sans SC"]


class TestDetection:
    def test_whitelist_order_wins(self, monkeypatch):
        monkeypatch.setattr(
            fc, "_family_supports_chinese", lambda name: name in {"Microsoft YaHei", "SimHei"}
        )
        monkeypatch.setattr(fc, "_chinese_families_from_system", lambda: ["Noto Sans SC"])

        assert fc._detect_font_win() == "Microsoft YaHei"

    def test_system_pool_covers_whitelist_miss(self, monkeypatch):
        monkeypatch.setattr(
            fc, "_family_supports_chinese", lambda name: name == "Noto Sans SC"
        )
        monkeypatch.setattr(
            fc, "_chinese_families_from_system", lambda: ["SimSun", "Noto Sans SC"]
        )

        assert fc._detect_font_win() == "Noto Sans SC"

    def test_system_pool_prefers_sans_over_serif(self, monkeypatch):
        fake = ["FangSong", "SimSun", "Noto Sans SC", "KaiTi", "Microsoft YaHei"]
        monkeypatch.setattr(
            QFontDatabase, "families", staticmethod(lambda writing_system=None: fake)
        )

        ranked = fc._chinese_families_from_system()

        assert ranked[0] == "Microsoft YaHei"
        assert ranked[-1] in {"FangSong", "SimSun", "KaiTi"}


class TestCachedSelection:
    def test_valid_cache_skips_detection(self, monkeypatch):
        stub = _FontSettingsStub("Microsoft YaHei UI", fc.CACHE_VERSION)
        monkeypatch.setattr("src.core.settings.AppSettings", lambda: stub)
        monkeypatch.setattr(fc, "_family_supports_chinese", lambda name: True)

        def _should_not_detect():
            raise AssertionError("覆盖校验通过时不应重新检测")

        monkeypatch.setattr(fc, "_detect_font_win", _should_not_detect)

        assert fc.get_windows_chinese_font_cached() == "Microsoft YaHei UI"
        assert fc.resolved_chinese_family() == "Microsoft YaHei UI"

    def test_cached_name_without_glyphs_is_replaced(self, monkeypatch):
        stub = _FontSettingsStub("SimSun", fc.CACHE_VERSION)
        monkeypatch.setattr("src.core.settings.AppSettings", lambda: stub)
        monkeypatch.setattr(
            fc, "_family_supports_chinese", lambda name: name == "Microsoft YaHei UI"
        )
        monkeypatch.setattr(fc, "_detect_font_win", lambda: "Microsoft YaHei UI")

        assert fc.get_windows_chinese_font_cached() == "Microsoft YaHei UI"
        assert (stub.name, stub.version) == ("Microsoft YaHei UI", fc.CACHE_VERSION)

    def test_stale_version_redetects(self, monkeypatch):
        stub = _FontSettingsStub("Microsoft YaHei UI", fc.CACHE_VERSION - 1)
        monkeypatch.setattr("src.core.settings.AppSettings", lambda: stub)
        monkeypatch.setattr(fc, "_family_supports_chinese", lambda name: True)
        monkeypatch.setattr(fc, "_detect_font_win", lambda: "DengXian")

        assert fc.get_windows_chinese_font_cached() == "DengXian"
        assert (stub.name, stub.version) == ("DengXian", fc.CACHE_VERSION)

    def test_no_chinese_font_returns_empty(self, monkeypatch):
        stub = _FontSettingsStub()
        monkeypatch.setattr("src.core.settings.AppSettings", lambda: stub)
        monkeypatch.setattr(fc, "_detect_font_win", lambda: "")

        assert fc.get_windows_chinese_font_cached() == ""
        assert fc.resolved_chinese_family() == ""


class TestBuildFont:
    def test_applies_stack_and_logs(self, qapp, monkeypatch):
        monkeypatch.setattr(fc, "get_windows_chinese_font_cached", lambda: "Microsoft YaHei UI")
        logger = _DummyLogger()
        monkeypatch.setattr("src.core.logger.get_logger", lambda name: logger)

        font = fc.build_windows_ui_font()

        assert font.families()[0] == "Segoe UI Variable Text"
        assert "Microsoft YaHei UI" in font.families()
        assert font.pixelSize() == 12
        assert any("Microsoft YaHei UI" in line for line in logger.infos)

    def test_warns_when_no_chinese_font(self, qapp, monkeypatch):
        monkeypatch.setattr(fc, "get_windows_chinese_font_cached", lambda: "")
        logger = _DummyLogger()
        monkeypatch.setattr("src.core.logger.get_logger", lambda name: logger)

        fc.build_windows_ui_font()

        assert logger.warnings