"""路径工具函数

两类职责：
- 资源定位（``resource_path``）：打包环境下的 bundle 内相对资源。
- 跨平台写法（``normalize_input_path`` / ``to_windows`` / ``to_posix``
  / ``format_for_copy``）：外部输入路径的规范化与展示/复制风格转换。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from src.utils.platform_setup import get_bundle_dir


def resource_path(relative_path: str) -> Path:
    """获取资源文件路径

    开发环境：从项目根目录解析
    打包环境：使用统一的 bundle_dir 定位
    """
    return get_bundle_dir() / relative_path


# ---------------------------------------------------------------------------
# 跨平台写法：Windows 反斜杠 / POSIX 正斜杠 / UNC 网盘路径
# ---------------------------------------------------------------------------

#: 命令行里不包引号就会断词或被当特殊字符的集合。含 ``=``（用户实测文件名
#: ``..._1am=0.8_Map.csv``）、``%``（``_20%SOC_``）、以及 PowerShell 会展开的
#: ``$`` / 反引号 —— 后两者仍用双引号包（CMD 优先），PowerShell 下需手工改
#: 单引号，已在 tooltip 里注明。
_SHELL_SPECIAL_RE = re.compile(r"[\s&()#!^\"'%<>,;=$`*?]")

#: 写法风格
STYLE_NATIVE = "native"
STYLE_WINDOWS = "windows"
STYLE_POSIX = "posix"

#: 引号策略
QUOTE_AUTO = "auto"
QUOTE_ALWAYS = "always"
QUOTE_NEVER = "never"


def _native_abspath(path: str) -> str:
    """按**当前平台**绝对化 + 规范化。

    单独抽一层是为了可测：macOS/Linux 上也能用 ``ntpath`` 覆盖本函数，
    验证 Windows 分支（实测 ``ntpath.abspath('//cnn002/share/a b.csv')``
    → ``\\\\cnn002\\share\\a b.csv``，UNC 前缀能正确保留）。
    """
    return os.path.abspath(os.path.expanduser(path))


def normalize_input_path(raw) -> str:
    """把外部来源（文件对话框 / 拖拽 / 命令行）的路径规范成当前平台的绝对写法。

    背景（实测）：Windows 上从 Explorer 把文件**拖**进主窗口时，Qt 的
    ``QUrl.toLocalFile()`` 会把 UNC 网盘路径产出成 ``//host/share/x.csv``
    —— 正斜杠，且主机名被强制小写（实测往返不恒等）。这种串粘回 Windows
    会被 Shell 当 URL 交给浏览器（实测会跳到 Edge），只有 ``\\host\\share``
    才能跳转网盘；而 ``QFileDialog`` 给的是原生反斜杠，所以症状是
    “有时能用有时不能用”。

    规则：
    - ``~`` 展开为用户目录；相对路径按当前工作目录绝对化。
    - 折叠重复分隔符；Windows 上 ``/`` 一律转 ``\\``，**并保留 UNC 的两层
      前导反斜杠**（``os.path.abspath`` 内部的 normpath 正好做这件事）。
    - **POSIX 上不动 ``\\``**：反斜杠在 Linux/macOS 是合法文件名字符。
    - 幂等：多次调用结果不变（重复规范化不会把 UNC 越拼越长）。
    - 空串原样返回（``abspath("")`` 会返回工作目录，不能走）。

    已知损失：主机名大小写无法从 ``QUrl`` 还原（Qt 解析时就小写了）；
    Windows 主机名不区分大小写，不影响可用性。
    """
    path = str(raw or "").strip()
    if not path:
        return ""
    return _native_abspath(path)


def _unify(path: str) -> str:
    """转成“以 ``/`` 为分隔、重复分隔符已折叠”的中间形式，UNC 保留两层前导。

    先统一再分发，``//h/s`` 与 ``\\h\\s`` 两种输入都落到同一条路径。
    """
    s = str(path or "").strip()
    if not s:
        return ""
    if s.startswith(("//", "\\\\")):
        return "//" + re.sub(r"[\\/]{2,}", "/", s[2:])
    return re.sub(r"[\\/]{2,}", "/", s)


def to_windows(path: str) -> str:
    r"""任意写法 → Windows 写法（反斜杠，UNC 保留 ``\\host\share``）。

    在 macOS/Linux 上也要能产出 Windows 串（用户会把路径发给 Windows 侧），
    所以不依赖 ``os.sep``，按“两种分隔符都认”纯字符串处理。
    """
    return _unify(path).replace("/", "\\")


def to_posix(path: str) -> str:
    r"""任意写法 → POSIX 写法（正斜杠；UNC 为 ``//host/share``）。

    注：POSIX 下反斜杠本是合法文件名字符，本函数**仍会把它当分隔符** ——
    这是“给人看的写法转换”，不是“给当前系统打开的路径解析”；真需按本机
    语义解析请走 ``normalize_input_path``。
    """
    return _unify(path).replace("\\", "/")


def display_path(path: str, style: str = STYLE_NATIVE) -> str:
    """按风格输出路径；``native`` 走当前平台规范化（幂等，不抛异常）。"""
    s = str(path or "")
    if not s.strip():
        return s
    if style == STYLE_WINDOWS:
        return to_windows(s)
    if style == STYLE_POSIX:
        return to_posix(s)
    return normalize_input_path(s)


def needs_shell_quoting(path: str) -> bool:
    """含空白或 shell 特殊字符时，在 CMD / PowerShell 里必须包引号。"""
    return bool(path) and _SHELL_SPECIAL_RE.search(str(path)) is not None


def quote_for_shell(path: str, mode: str = QUOTE_AUTO) -> str:
    """按 CMD / PowerShell 习惯包双引号（与 Explorer「复制为路径」一致）。

    自身已含 ``"`` 时**不包**：CMD 与 PowerShell 的内嵌引号转义规则不兼容，
    猜错比不加更糟，交回用户处理。
    """
    s = str(path or "")
    if mode == QUOTE_NEVER or not s or '"' in s:
        return s
    if mode == QUOTE_AUTO and not needs_shell_quoting(s):
        return s
    return f'"{s}"'


def format_for_copy(path: str, style: str = STYLE_NATIVE, quote: str = QUOTE_AUTO) -> str:
    """剪贴板的唯一出口：先转写法风格，再按策略包引号。"""
    return quote_for_shell(display_path(path, style), quote)
