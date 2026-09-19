"""P2-14：字体族查询要进程级记忆一次，缓存命中路径不能再枚举系统字体。

`QFontDatabase.families()` 在 Windows 上要遍历全部已安装字体（模块头自述
50-100 ms）。旧写法把它放在**缓存命中**的校验上（`cached_name in families()`），
等于把缓存本该省下的时间原样花回去；splash 的 `draw_text` 更是在每帧绘制里
枚举一次。现在三处都改问 `installed_font_families()`，一次运行只枚举一次。
"""

from __future__ import annotations

import inspect

import pytest

from PySide6.QtGui import QFontDatabase

from src.core import font_cache as fc
from src.ui import splash_screen


FAKE_FAMILIES = ["Alpha Font", "Menlo", "Microsoft YaHei"]
# Menlo 取自 _MONO_FONT_PRIORITY：失效回落路径要能检出它


@pytest.fixture()
def enumerate_once(monkeypatch):
    """替身 `families()`：数它被调了几次。"""
    monkeypatch.setattr(fc, "_installed_families", None)
    calls: list[int] = []

    def fake_families():
        calls.append(1)
        return list(FAKE_FAMILIES)

    monkeypatch.setattr(QFontDatabase, "families", staticmethod(fake_families))
    return calls


class _StubSettings:
    def __init__(self, name, version):
        self._name = name
        self._version = version

    def get_mono_font_cache_version(self):
        return self._version

    def get_mono_font_name(self):
        return self._name

    def set_mono_font_cache_version(self, v):
        self._version = v

    def set_mono_font_name(self, n):
        self._name = n


class TestMemoizedFamilyLookup:
    def test_enumerates_only_once_across_calls(self, enumerate_once):
        assert fc.installed_font_families() == frozenset(FAKE_FAMILIES)
        for _ in range(5):
            fc.installed_font_families()

        assert len(enumerate_once) == 1

    def test_cache_hit_path_does_not_re_enumerate(self, enumerate_once, monkeypatch):
        """「缓存命中」本该是最便宜的一条路，旧写法每次都要遍历一遍系统字体。"""
        monkeypatch.setattr(
            "src.core.settings.AppSettings",
            lambda: _StubSettings("Alpha Font", fc._MONO_CACHE_VERSION),
        )

        assert fc.get_monospace_font_cached() == "Alpha Font"
        assert fc.get_monospace_font_cached() == "Alpha Font"

        assert len(enumerate_once) == 1, f"命中缓存仍枚举了 {len(enumerate_once)} 次"

    def test_stale_cache_still_falls_back_to_detection(self, enumerate_once, monkeypatch):
        """记忆不能把「缓存字体已被卸载」的失效路径一起抹掉。"""
        monkeypatch.setattr(
            "src.core.settings.AppSettings",
            lambda: _StubSettings("Gone Font", fc._MONO_CACHE_VERSION),
        )

        detected = fc.get_monospace_font_cached()

        assert detected in FAKE_FAMILIES, f"应回落到现场检测，实得 {detected!r}"

    def test_detector_shares_the_same_snapshot(self, enumerate_once):
        assert fc._detect_font(["Nope", "Microsoft YaHei"]) == "Microsoft YaHei"
        assert fc._detect_font(["Alpha Font"]) == "Alpha Font"

        assert len(enumerate_once) == 1


class TestSplashPaintPath:
    def test_paint_path_no_longer_touches_qfontdatabase(self):
        """绘制路径里出现 QFontDatabase 就是「每帧枚举」，源码层直接钉死。"""
        source = inspect.getsource(splash_screen)

        assert "QFontDatabase" not in source
        assert "installed_font_families()" in source
