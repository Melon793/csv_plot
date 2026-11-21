"""第 6 轮批次 D（P2 精选）：菜单/对话框泄漏 AST 护栏 / shelf tooltip 转义与
暗色跟随 / 首色决策护栏 / count_template_variables 畸形守卫。

E-1..E-4 + 同族三处：QMenu/QMessageBox 实例 exec 后必须 deleteLater
（带 parent 的 C++ 对象不随 Python 包装器释放，长寿命宿主下累积）。
P2-50/51：shelf tooltip 转义 + palette 跟随；B-2：style_summary 含预设。
C-4：首色 #1658DB 决策护栏（hex + 白底对比度 ≥4.5）。
P2-48：count_template_variables 对 unhashable/null curves 元素不抛。
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from src.core.config import DEFAULT_CURVE_COLORS
from src.core.template_models import count_template_variables


def test_all_menu_exec_sites_release():
    """AST 护栏：构造 QMenu/QMessageBox 且调用 .exec 的函数必须有 deleteLater。"""
    src_root = Path(__file__).resolve().parents[2] / "src"
    offenders = []
    for path in sorted(src_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            constructs = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id in ("QMenu", "QMessageBox")
                for n in ast.walk(node)
            )
            has_exec = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "exec"
                for n in ast.walk(node)
            )
            has_release = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "deleteLater"
                for n in ast.walk(node)
            )
            if constructs and has_exec and not has_release:
                offenders.append(f"{path.name}:{node.lineno} {node.name}")
    assert not offenders, f"菜单/对话框 exec 位点缺 deleteLater：{offenders}"


def test_shelf_tooltip_escaped_and_palette_follows(qapp):
    from PySide6.QtGui import QColor, QPalette
    from src.ui.widgets.curve_shelf import CurveShelfWidget, ShelfEntry

    shelf = CurveShelfWidget()
    try:
        entry = ShelfEntry(key="k1", var_name="<b>x</b>&y", line_preset="bold")
        shelf.add_entry(entry)
        item = shelf.item(0)
        tip = item.toolTip()
        assert "&lt;b&gt;" in tip and "<b>x</b>" not in tip, (
            f"变量名尖括号必须转义（防 QToolTip 当标签吃掉）：{tip}"
        )
        assert "<br>" in tip, "换行必须以 <br> 还原（<qt> 富文本口径）"
        assert "预设：bold" in tip, "style_summary 必须含预设（B-2）"
        # 评审 NB1：非 stale 前景 brush 清空 → delegate 走实时 palette
        # （运行期暗色切换自愈，且不覆盖选中态 HighlightedText）
        from PySide6.QtCore import Qt as _Qt
        assert item.data(_Qt.ItemDataRole.ForegroundRole) is None, (
            "非 stale 条目不得钉死前景 brush"
        )
        # stale 条目用 Disabled Text（暗色可读）
        dark = QPalette()
        dark.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                      QColor("#808080"))
        shelf.setPalette(dark)
        entry.stale = True
        shelf._refresh_item(entry)
        assert item.foreground().color().name() == "#808080", (
            f"stale 前景须跟随 Disabled Text：{item.foreground().color().name()}"
        )
    finally:
        shelf.deleteLater()


def test_first_color_decision_guard():
    """C-4：首色钴蓝决策护栏——hex 钉住 + 白底 WCAG 对比度 ≥4.5。"""
    first = DEFAULT_CURVE_COLORS[0]
    assert first.upper() == "#1658DB", f"首色决策变更须显式更新本护栏：{first}"

    def luminance(hex_color: str) -> float:
        r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        chan = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
                for c in (r, g, b)]
        return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]

    ratio = (1.0 + 0.05) / (luminance(first) + 0.05)
    assert ratio >= 4.5, f"白底对比度须 ≥4.5（WCAG AA）：{ratio:.2f}"


@pytest.mark.parametrize("curves", [
    [["unhashable"], None, 123, "ok"],
    "scalar-not-list",
    None,
])
def test_count_template_variables_tolerates_malformed(curves):
    config = {"pages": [{"plots": [{"curves": curves}]}]}
    n = count_template_variables(config)  # 不得抛 TypeError
    assert isinstance(n, int)


def test_count_template_variables_normal_unchanged():
    config = {"pages": [{"plots": [
        {"curves": ["a", "b"]}, {"curves": ["b", "c"]},
    ]}]}
    assert count_template_variables(config) == 3


def test_count_template_variables_plots_scalar_tolerated():
    """第 6 轮 NB4：plots 标量（手写畸形档）不得抛 TypeError。"""
    assert count_template_variables({"pages": [{"plots": 5}]}) == 0
    assert count_template_variables({"plots": "abc"}) == 0