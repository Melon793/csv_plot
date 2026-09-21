"""``reveal_in_file_manager`` 单元测试：钉「交给系统的那一串」的形状。

Windows 那一支刻意传 str 而不是 argv 列表，两者只差一处引号的**位置**，而位置
错了就是"打开我的文档"（实测共享盘目录下 8/8 中招，含空格、``&``、``%``、``=``、
中文与全角标点）。形状是纯字符串问题，后果却只有真机看得出来，所以在这里把形状
钉死，而不是留一条"看起来调用了系统命令"的断言。

真实主机名/共享名/客户名一律不出现：路径由 ``tmp_path`` 现造，文件名全是虚构
标识符，请勿改回现场真名。
"""

from __future__ import annotations

import subprocess

import pytest

from src.ui.widgets import status_drawer as sd

#: 现场会出现的字符：空格、&、%、=、$、#、()、[]、中文、全角空格与全角逗号。
#: 不含 Windows 文件名非法字符 " \\ / : | ? *
NASTY_NAMES = [
    "plain.csv",
    "a b.csv",
    "map&1am.csv",
    "20%SOC_map.csv",
    "p=0.8_map.csv",
    "a&b c(1)[$]#x.csv",
    "标定_映射 ENG01 & 复测(2).csv",
    "全角　符号，测试.csv",
]


@pytest.fixture
def popen_calls(monkeypatch):
    """截获交给系统的参数，不真启动任何文件管理器。"""
    calls: list = []
    monkeypatch.setattr(sd.subprocess, "Popen", lambda args, **kw: calls.append(args))
    return calls


def _make(tmp_path, name: str) -> tuple[str, str]:
    """造一个真实文件，返回 (文件全路径, 所在文件夹)。

    用真文件而不是假字符串：函数开头有 ``os.path.exists`` 守卫，造假路径就得连
    ``os.path`` 一起 patch，测试会跟着实现一起失去意义。
    """
    f = tmp_path / name
    f.write_text("x\n", encoding="utf-8")
    return str(f), str(tmp_path)


# -- Windows：引号只能在逗号之后 --------------------------------------------


@pytest.mark.parametrize("name", NASTY_NAMES)
def test_win32_wraps_only_the_path(name, tmp_path, monkeypatch, popen_calls):
    """怪名字逐个验：整串必须是 ``explorer /select,"路径"``，路径原样不转义。"""
    path, folder = _make(tmp_path, name)
    monkeypatch.setattr(sd.sys, "platform", "win32")

    assert sd.reveal_in_file_manager(path, folder) == ""
    assert popen_calls == [f'explorer /select,"{path}"']


def test_win32_select_verb_is_never_inside_the_quotes(tmp_path, monkeypatch, popen_calls):
    """回归点单独钉一条：``"/select`` 这个子串一出现，就是"打开我的文档"。"""
    path, folder = _make(tmp_path, "a b.csv")
    monkeypatch.setattr(sd.sys, "platform", "win32")

    sd.reveal_in_file_manager(path, folder)
    assert '"/select' not in popen_calls[0]


def test_list_form_is_what_breaks_on_windows(tmp_path):
    """反证：一旦有人把 win32 支改回 list，``list2cmdline`` 就产出被禁的那种串。

    这条不依赖平台，在 macOS 上跑同样成立，所以它是上面两条跨平台的兜底。
    """
    path = str(tmp_path / "a b.csv")
    as_list = subprocess.list2cmdline(["explorer", f"/select,{path}"])

    assert '"/select' in as_list, "list2cmdline 的包引号规则变了，上面的守卫要重估"


def test_win32_folder_only_quotes_the_folder(tmp_path, monkeypatch, popen_calls):
    """只有目录、没有文件时：``explorer "目录"``，不带 /select。"""
    _, folder = _make(tmp_path, "plain.csv")
    monkeypatch.setattr(sd.sys, "platform", "win32")

    assert sd.reveal_in_file_manager("", folder) == ""
    assert popen_calls == [f'explorer "{folder}"']


# -- 另两平台仍走 argv 列表 --------------------------------------------------


def test_darwin_still_uses_argv_list(tmp_path, monkeypatch, popen_calls):
    path, folder = _make(tmp_path, "a b.csv")
    monkeypatch.setattr(sd.sys, "platform", "darwin")

    assert sd.reveal_in_file_manager(path, folder) == ""
    assert popen_calls == [["open", "-R", path]]


def test_linux_still_uses_argv_list(tmp_path, monkeypatch, popen_calls):
    path, folder = _make(tmp_path, "a b.csv")
    monkeypatch.setattr(sd.sys, "platform", "linux")

    assert sd.reveal_in_file_manager(path, folder) == ""
    assert popen_calls == [["xdg-open", folder]]


# -- 交不出去的两支 ----------------------------------------------------------


def test_missing_file_launches_nothing_and_says_why(monkeypatch, popen_calls):
    monkeypatch.setattr(sd.sys, "platform", "win32")

    assert sd.reveal_in_file_manager("/nowhere/a b.csv", "/nowhere") == "文件已不在原位置"
    assert popen_calls == []


def test_nothing_to_open_launches_nothing(monkeypatch, popen_calls):
    monkeypatch.setattr(sd.sys, "platform", "win32")

    assert sd.reveal_in_file_manager("", "") == "没有可打开的路径"
    assert popen_calls == []
