"""LayoutGridSelector 视觉状态 component 测试（offscreen）。

覆盖点：
1. 三态渲染（新布局预览 sel / 当前布局 current / 未选 off）与回退；
2. 单元格圆角恒为独立圆角片（已取消同色区合并的方角接缝）；
3. 渲染签名缓存（重复刷新不重设样式，反向验证性能意图）；
4. 网格几何（行列间距、不重叠）与尺寸约束（不锁小、hover 不抖动）；
5. 提示区双层文案格式、行列标尺、accessibleName；
6. values() 接口与方向键起点等既有语义未回归。
"""

import pytest

from PySide6.QtCore import QCoreApplication, QEvent, QSize, Qt
from PySide6.QtGui import QFocusEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QDialog, QLabel, QLayout

from src.ui.dialogs.layout_grid_selector import (
    LayoutGridSelector,
    _CELL_SIZE,
    _CELL_SPACING,
    _R_CELL,
    _RULER_THICK,
    _S_CURRENT,
    _S_OFF,
    _S_SEL,
)


@pytest.fixture()
def selector(qtbot):
    """4x3 网格、当前布局 3x1 的选择器实例（不 show，纯状态断言）

    交给 qtbot 登记，收尾自动 close + deleteLater，避免顶层窗口跨用例存活。
    """
    dlg = LayoutGridSelector(max_rows=4, max_cols=3, cur_rows=3, cur_cols=1)
    qtbot.addWidget(dlg)
    return dlg


def _state(dlg, r, c):
    return dlg.cells[r][c].visual


# ---------- 三态渲染 ----------

def test_grid_shape_matches_max_rows_cols(selector):
    assert len(selector.cells) == 4
    assert all(len(row) == 3 for row in selector.cells)


def test_initial_state_marks_current_layout_only(selector):
    """未悬停时：当前布局 3x1 显示为 current，其余 off（不再有单一高亮态）"""
    for r in range(3):
        assert _state(selector, r, 0) == _S_CURRENT, f"row {r} col 0"
    assert _state(selector, 3, 0) == _S_OFF
    assert _state(selector, 0, 1) == _S_OFF
    assert _state(selector, 3, 2) == _S_OFF


def test_hover_creates_sel_rect_and_keeps_current_band(selector):
    """悬停 (1,1)：2x2 为 sel；sel 之外的当前布局格仍为 current，新旧同框可比"""
    selector._on_cell_hovered(1, 1)
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert _state(selector, r, c) == _S_SEL, f"({r},{c})"
    assert _state(selector, 2, 0) == _S_CURRENT
    assert _state(selector, 3, 0) == _S_OFF
    assert _state(selector, 0, 2) == _S_OFF


def test_grid_left_falls_back_to_initial_states(selector):
    """离开网格回退到初始 current 态，而非全部灰"""
    selector._on_cell_hovered(3, 2)
    selector._on_grid_left()
    assert _state(selector, 2, 0) == _S_CURRENT
    assert _state(selector, 3, 2) == _S_OFF
    assert _state(selector, 0, 1) == _S_OFF


# ---------- 圆角（恒为独立圆角片）----------

def test_every_cell_keeps_independent_radius_in_all_states(selector):
    """三态下每格四角都恒为 _R_CELL，不得再出现同色区合并的方角接缝

    回归护栏：接缝逻辑（逐角 0px）已整块删除，若被重新引入会在样式串里
    留下 border-*-radius 写法，本用例按样式串内容捕获。
    """
    for hover in [None, (1, 1), (3, 2), (0, 0)]:
        if hover is None:
            selector._on_grid_left()
        else:
            selector._on_cell_hovered(*hover)
        for r in range(selector.max_rows):
            for c in range(selector.max_cols):
                css = selector.cells[r][c].styleSheet()
                assert f"border-radius: {_R_CELL}px;" in css, f"({r},{c}) @ {hover}"
                assert "border-top-left-radius" not in css


# ---------- 签名缓存 ----------

def test_repeated_highlight_skips_restyle(selector):
    """同状态重复刷新不得重设样式串（鼠标每划一格都会触发整网格刷新）"""
    counter = {"n": 0}
    for row in selector.cells:
        for cell in row:
            orig = cell.setStyleSheet

            def wrapped(style, _orig=orig, _counter=counter):
                _counter["n"] += 1
                return _orig(style)

            cell.setStyleSheet = wrapped

    selector._update_highlight(1, 1)
    first = counter["n"]
    assert first > 0, "首次刷新应发生样式重设"

    selector._update_highlight(1, 1)
    assert counter["n"] == first, "签名未变时不应再重设样式"

    selector._update_highlight(2, 2)
    assert counter["n"] > first, "状态变化时应重设受影响单元格"


# ---------- 网格几何（不重叠） ----------

def _expected_grid_size(n_rows, n_cols):
    """标尺行/列 + n 个单元格 + n 段间距（含标尺与首格之间的一段）"""
    return QSize(
        _RULER_THICK + n_cols * _CELL_SIZE + n_cols * _CELL_SPACING,
        _RULER_THICK + n_rows * _CELL_SIZE + n_rows * _CELL_SPACING,
    )


def test_grid_row_col_pins_are_applied(selector):
    """行列尺寸被显式钉住：标尺带 = _RULER_THICK，单元格行/列 = _CELL_SIZE

    回归护栏：不显式锁定时，cocoa 上构造期（未 polish）的 QGridLayout 会漏算
    末行（137x185 vs 145x188），使整体尺寸算小、方块纵向重叠。
    """
    gl = selector.grid_widget.layout()
    assert gl.columnMinimumWidth(0) == _RULER_THICK
    assert gl.rowMinimumHeight(0) == _RULER_THICK
    assert [
        gl.columnMinimumWidth(i + 1) for i in range(selector.max_cols)
    ] == [_CELL_SIZE] * selector.max_cols
    assert [
        gl.rowMinimumHeight(i + 1) for i in range(selector.max_rows)
    ] == [_CELL_SIZE] * selector.max_rows


def test_grid_size_hint_covers_formula(selector):
    """网格 sizeHint 不得小于解析公式值。

    只断言下限：cocoa 上 mac 样式会给 QGridLayout 额外纵向留白（实测多 8px，
    与标尺子控件类型无关，不放标尺行也一样多），精确等式仅 offscreen 成立。
    实际不重叠由 test_adjacent_cells_do_not_overlap 按几何断言，两平台均成立。
    """
    expect = _expected_grid_size(4, 3)
    hint = selector.grid_widget.sizeHint()
    assert hint.width() >= expect.width()
    assert hint.height() >= expect.height()


def test_cell_style_sheet_carries_radius(selector):
    """样式串里必须真的带上了圆角与渐变

    QSS 靠动态属性选择器生效，属性名/设置顺序被改坏时样式会整体退化为
    默认按钮而 Python 侧签名断言仍全绿，故补一条渲染层弱断言。
    """
    selector._on_cell_hovered(1, 1)
    css = selector.cells[0][0].styleSheet()
    assert 'QPushButton[cellState="sel"]' in css
    assert "qlineargradient" in css
    assert f"border-radius: {_R_CELL}px;" in css


def test_size_is_owned_by_layout_constraint(selector):
    """尺寸交给 QLayout.SetFixedSize，不得在构造期用 sizeHint 锁死。

    回归护栏：曾用 setFixedSize(self.sizeHint())，而构造期 hint 在 polish 后
    会变（实测 offscreen 少 2px 导致压叠、cocoa 多 8px 导致留白）。
    """
    assert selector.layout().sizeConstraint() == QLayout.SizeConstraint.SetFixedSize


def test_dialog_not_locked_smaller_than_layout_minimum(selector, qtbot):
    """窗口尺寸不得小于布局所需最小尺寸（否则固定尺寸方块会被压叠）"""
    selector.show()
    qtbot.waitExposed(selector)
    need = selector.layout().minimumSize()
    assert selector.width() >= need.width()
    assert selector.height() >= need.height()


def test_window_size_stable_across_hover(selector, qtbot):
    """遍历全部 hover 预览，窗口尺寸不得随提示文案变长变短而抖动"""
    selector.show()
    qtbot.waitExposed(selector)
    base = (selector.width(), selector.height())
    for r in range(selector.max_rows):
        for c in range(selector.max_cols):
            selector._on_cell_hovered(r, c)
            qtbot.wait(0)
            assert (selector.width(), selector.height()) == base, f"hover({r},{c}) 尺寸抖动"
    selector._on_grid_left()
    qtbot.wait(0)
    assert (selector.width(), selector.height()) == base


def test_adjacent_cells_do_not_overlap(selector, qtbot):
    """show 后任意两个单元格矩形不得相交，且行/列间距为设计值

    这是 macOS 方块重叠缺陷的直接护栏；cocoa 与 offscreen 下均应成立。
    """
    selector.show()
    qtbot.waitExposed(selector)
    rects = [c.geometry() for row in selector.cells for c in row]
    for i, a in enumerate(rects):
        for b in rects[i + 1:]:
            assert not a.intersects(b), f"{a} 与 {b} 重叠"
    pitch = _CELL_SIZE + _CELL_SPACING
    for r in range(selector.max_rows - 1):
        for c in range(selector.max_cols):
            assert selector.cells[r + 1][c].y() - selector.cells[r][c].y() == pitch
    for r in range(selector.max_rows):
        for c in range(selector.max_cols - 1):
            assert selector.cells[r][c + 1].x() - selector.cells[r][c].x() == pitch


# ---------- 事件路径（信号连接不得静默断开）----------

def test_real_enter_event_drives_preview(selector, qtbot):
    """走真实事件：CellButton.enterEvent -> cell_hovered -> 预览刷新"""
    selector.show()
    qtbot.waitExposed(selector)
    QCoreApplication.sendEvent(selector.cells[2][1], QEvent(QEvent.Type.Enter))
    assert selector.hint_label.text() == "3 × 2"
    assert _state(selector, 1, 1) == _S_SEL


def test_real_leave_event_falls_back(selector, qtbot):
    """走真实事件：容器 leaveEvent -> grid_left -> 回退初始高亮"""
    selector.show()
    qtbot.waitExposed(selector)
    QCoreApplication.sendEvent(selector.cells[2][1], QEvent(QEvent.Type.Enter))
    QCoreApplication.sendEvent(selector.grid_widget, QEvent(QEvent.Type.Leave))
    assert selector.hint_label.text() == "3 × 1"
    assert _state(selector, 1, 1) == _S_OFF


def test_real_focus_out_falls_back(selector, qtbot):
    """走真实事件：焦点离开网格（如 Tab 到取消按钮）同样回退"""
    selector.show()
    qtbot.waitExposed(selector)
    QCoreApplication.sendEvent(selector.cells[2][1], QEvent(QEvent.Type.Enter))
    QCoreApplication.sendEvent(
        selector.grid_widget, QFocusEvent(QEvent.Type.FocusOut)
    )
    assert selector.hint_label.text() == "3 × 1"


def test_real_mouse_press_confirms(selector, qtbot):
    """走真实事件：mousePressEvent -> cell_clicked -> accept 并记录尺寸"""
    selector.show()
    qtbot.waitExposed(selector)
    QTest.mouseClick(selector.cells[2][1], Qt.MouseButton.LeftButton)
    assert selector.values() == (3, 2)
    assert selector.result() == int(QDialog.DialogCode.Accepted)


# ---------- 提示区文案 ----------

def test_hint_texts_initial_and_hover(selector):
    assert selector.hint_label.text() == "3 × 1"
    assert selector.hint_sub_label.text() == "当前布局 · 点击方块即可切换"

    selector._on_cell_hovered(1, 1)
    assert selector.hint_label.text() == "2 × 2"
    assert "当前 3 × 1" in selector.hint_sub_label.text()

    selector._on_grid_left()
    assert selector.hint_label.text() == "3 × 1"


# ---------- 标尺与无障碍 ----------

def test_ruler_labels_present(selector):
    """第 0 行列号 + 第 0 列行号，共 max_rows + max_cols 个数字标签"""
    labels = [
        w.text()
        for w in selector.grid_widget.findChildren(QLabel)
        if w.text().isdigit()
    ]
    assert sorted(labels) == sorted(
        ["1", "2", "3", "4"] + ["1", "2", "3"]
    )


def test_cell_accessible_name(selector):
    assert selector.cells[2][1].accessibleName() == "3行2列"


# ---------- 既有接口语义（回归红线）----------

def test_values_default_to_current_layout(selector):
    assert selector.values() == (3, 1)


def test_click_returns_selected_shape(selector):
    selector._on_cell_clicked(2, 1)
    assert selector.values() == (3, 2)
    assert selector.result() == int(QDialog.DialogCode.Accepted)


def test_key_move_starts_from_current_layout(selector):
    """方向键从初始 current 位置 (3,1) 出发，仅预览不确认"""
    selector._on_key_move(-1, 0)
    assert selector.hint_label.text() == "2 × 1"
    assert _state(selector, 1, 0) == _S_SEL
    # 第 3 行已不在预览范围内，但仍属于当前布局 -> 回退为淡蓝 current
    assert _state(selector, 2, 0) == _S_CURRENT
    assert selector.result() == 0
