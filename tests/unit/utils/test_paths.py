"""utils.paths 单元测试：资源路径解析 + 跨平台写法转换。"""

from __future__ import annotations

import ntpath
import os
from pathlib import Path

import pytest

from src.utils import paths
from src.utils.paths import (
    QUOTE_ALWAYS,
    QUOTE_AUTO,
    QUOTE_NEVER,
    STYLE_NATIVE,
    STYLE_POSIX,
    STYLE_WINDOWS,
    display_path,
    format_for_copy,
    needs_shell_quoting,
    normalize_input_path,
    quote_for_shell,
    resource_path,
    to_posix,
    to_windows,
)

#: 用户实测报障的原始串（Windows 拖拽后 Qt 产出的形态）：正斜杠 UNC + 空格 + ``=``
UNC_POSIX = (
    "//fileserver/team-share/PRJ-0000-00_Demo_ENG01_TC01_ EU7 Calibration"
    "/b_Technics/c_Testing/04_Test_Result/OBM_Mapping"
    "/20260101_DEMO_ENG01_1am=0.8_Map/20260101_DEMO_ENG01_1am=0.8_Map.csv"
)
UNC_WIN = UNC_POSIX.replace("/", "\\")


def test_resource_path_joins_relative():
    p = resource_path("assets/icon.png")
    assert isinstance(p, Path)
    assert p.as_posix().endswith("assets/icon.png")


def test_existing_asset_resolves():
    """开发环境下项目自带的 assets 应可解析到存在的文件"""
    p = resource_path("assets/icon.png")
    assert p.exists()


# ---------------------------------------------------------------------------
# 写法转换（不依赖当前平台，纯字符串）
# ---------------------------------------------------------------------------


class TestStyleConversion:
    def test_posix_unc_becomes_windows_unc(self):
        """报障场景：'//host/share' 必须变成 '\\host\\share'，否则 Windows 当 URL"""
        out = to_windows(UNC_POSIX)
        assert out.startswith("\\\\fileserver\\")
        assert "/" not in out
        assert out == UNC_WIN

    def test_windows_unc_is_left_alone(self):
        assert to_windows(UNC_WIN) == UNC_WIN

    def test_drive_letter_slash_converted(self):
        assert to_windows("D:/Messung/a b/x.dat") == r"D:\Messung\a b\x.dat"

    def test_repeated_separators_collapsed_but_unc_prefix_kept(self):
        assert to_windows("//h//s///a.csv") == r"\\h\s\a.csv"
        assert to_posix("\\\\h\\\\s\\\\\\a.csv") == "//h/s/a.csv"

    def test_local_posix_absolute_unchanged(self):
        assert to_posix("/Users/x/a b/c.csv") == "/Users/x/a b/c.csv"

    def test_conversions_idempotent(self):
        for src in (UNC_POSIX, UNC_WIN, "D:/x/y.csv", r"C:\a\b", "/tmp/x"):
            assert to_windows(to_windows(src)) == to_windows(src)
            assert to_posix(to_posix(src)) == to_posix(src)

    def test_windows_and_posix_are_mutual_inverse_on_unc(self):
        assert to_posix(to_windows(UNC_POSIX)) == UNC_POSIX

    def test_empty_and_none_safe(self):
        assert to_windows("") == "" and to_posix("") == ""
        assert to_windows(None) == "" and to_posix(None) == ""

    def test_display_path_style_dispatch(self):
        assert display_path(UNC_POSIX, STYLE_WINDOWS) == UNC_WIN
        assert display_path(UNC_WIN, STYLE_POSIX) == UNC_POSIX
        assert display_path(UNC_WIN, "不认识的风格") == normalize_input_path(UNC_WIN)
        assert display_path("  ", STYLE_WINDOWS) == "  "


# ---------------------------------------------------------------------------
# 入口规范化（当前平台语义）
# ---------------------------------------------------------------------------


class TestNormalizeInputPath:
    def test_windows_unc_normalized(self, monkeypatch):
        """Windows 分支在 macOS 上也能验：直接把底层换成 ntpath（纯 Python）。

        这是本功能的根因用例：拖拽给的 '//fileserver/share/a b.csv' 经
        ``os.path.abspath`` 必须得到 '\\\\fileserver\\\\share\\\\a b.csv'。
        """
        monkeypatch.setattr(paths, "_native_abspath", ntpath.abspath)
        out = normalize_input_path("//fileserver/share/a b.csv")
        assert out == "\\\\fileserver\\share\\a b.csv"
        assert normalize_input_path(out) == out  # 幂等

    def test_windows_relative_gets_cwd_and_drive(self, monkeypatch):
        monkeypatch.setattr(paths, "_native_abspath", lambda p: "D:\\cwd\\" + p.replace("/", "\\"))
        assert normalize_input_path("data/x.csv") == r"D:\cwd\data\x.csv"

    @pytest.mark.skipif(os.sep != "/", reason="仅 POSIX：反斜杠是合法文件名字符")
    def test_posix_does_not_treat_backslash_as_separator(self):
        assert normalize_input_path("/tmp/a\\b.csv") == "/tmp/a\\b.csv"

    def test_expanduser(self):
        out = normalize_input_path("~/some_file.csv")
        assert not out.startswith("~")
        assert out.endswith("some_file.csv")

    def test_relative_to_absolute(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert normalize_input_path("sub/x.csv") == str(tmp_path / "sub" / "x.csv")

    def test_idempotent_on_native_input(self, tmp_path):
        once = normalize_input_path(str(tmp_path / "a b.csv"))
        assert normalize_input_path(once) == once

    def test_empty_none_and_blank(self):
        assert normalize_input_path("") == ""
        assert normalize_input_path(None) == ""
        assert normalize_input_path("   ") == ""


# ---------------------------------------------------------------------------
# 命令行引号
# ---------------------------------------------------------------------------


class TestShellQuoting:
    @pytest.mark.parametrize(
        "path",
        [
            r"\\fileserver\share\a b.csv",  # 空格（用户实测路径）
            r"\\fileserver\share\1am=0.8.csv",  # 等号
            r"D:\data\_20%SOC_.mf4",  # 百分号
            r"D:\a&b\x.csv",
            r"D:\a(b)\x.csv",
            r"D:\p\$env.csv",  # PowerShell 会展开
            r"D:\a#1\x.csv",
        ],
    )
    def test_special_chars_need_quoting(self, path):
        assert needs_shell_quoting(path)

    @pytest.mark.parametrize("path", [r"D:\data\x.csv", "/tmp/x.csv", ""])
    def test_plain_paths_need_no_quoting(self, path):
        assert not needs_shell_quoting(path)

    def test_auto_quotes_only_when_needed(self):
        assert quote_for_shell(r"D:\a b\x.csv") == '"D:\\a b\\x.csv"'
        assert quote_for_shell(r"D:\ab\x.csv") == r"D:\ab\x.csv"

    def test_always_and_never(self):
        assert quote_for_shell(r"D:\ab\x.csv", QUOTE_ALWAYS) == '"' + r"D:\ab\x.csv" + '"'
        assert quote_for_shell(r"D:\a b\x.csv", QUOTE_NEVER) == r"D:\a b\x.csv"

    def test_unknown_mode_falls_back_to_auto(self):
        """配置里把引号策略写错 → 当 auto 用，不得静默变成“永远包引号”。"""
        assert quote_for_shell(r"D:\ab\x.csv", "win") == r"D:\ab\x.csv"
        assert quote_for_shell(r"D:\a b\x.csv", "win") == '"D:\\a b\\x.csv"'

    def test_path_containing_double_quote_is_not_wrapped(self):
        weird = r'D:\a b\x"y.csv'
        assert quote_for_shell(weird, QUOTE_ALWAYS) == weird

    def test_format_for_copy_combines_style_and_quote(self):
        assert format_for_copy(UNC_POSIX, STYLE_WINDOWS, QUOTE_AUTO) == f'"{UNC_WIN}"'
        assert (
            format_for_copy(UNC_WIN, STYLE_POSIX, QUOTE_NEVER)
            == "//fileserver/team-share/PRJ-0000-00_Demo_ENG01_TC01_ EU7 Calibration"
            "/b_Technics/c_Testing/04_Test_Result/OBM_Mapping"
            "/20260101_DEMO_ENG01_1am=0.8_Map/20260101_DEMO_ENG01_1am=0.8_Map.csv"
        )
        # 无特殊字符时 auto 不包引号（粘 Excel 干净）
        assert format_for_copy("D:/data/x.csv", STYLE_WINDOWS, QUOTE_AUTO) == r"D:\data\x.csv"
        assert STYLE_NATIVE == "native"
