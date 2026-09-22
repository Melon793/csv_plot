"""e2e 加载与绘图链路测试：模拟 help.md 的用户旅程。

链路：导入数据文件（按钮 + 文件对话框替身）→ 变量列表刷新 →
拖拽变量到绘图区 → 曲线出现 → Ctrl+R 切换光标。

对应 help.md：
1️⃣ 加载数据（点击「导入数据文件」按钮）
2️⃣ 浏览数据（变量列表出现所有变量）
3️⃣ 绘制图表（从变量列表拖拽变量到绘图区域）
4️⃣ 交互操作（Ctrl+R 光标切换、Ctrl+Y 仅调节 Y 轴）
"""

import pytest

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QDropEvent

from src.ui.drag_drop import build_var_mimedata

# 保活容器：offscreen 下手工构造的 QMimeData 需防 GC（陷阱 #2）
_keep_alive: list = []


@pytest.mark.full_matrix
def test_default_matrix_builds_full_grid(loaded_window):
    """默认布局按 4×3 建满矩阵，并只显示默认可见区（3×1）。

    e2e 其余用例由 conftest 的 `_small_plot_matrix` 替身压到 1×1 提速
    （每例省约 95ms）；本用例以 `full_matrix` marker 跳过替身，把原先
    "54 例顺带覆盖"的建满矩阵行为显式锁定。
    """
    from src.core.config import (
        PLOT_COL_CURRENT_DEFAULT,
        PLOT_COL_MAX_DEFAULT,
        PLOT_ROW_CURRENT_DEFAULT,
        PLOT_ROW_MAX_DEFAULT,
    )

    mw = loaded_window
    assert (PLOT_ROW_MAX_DEFAULT, PLOT_COL_MAX_DEFAULT) == (4, 3)
    assert len(mw.plot_widgets) == PLOT_ROW_MAX_DEFAULT * PLOT_COL_MAX_DEFAULT == 12
    visible = [c for c in mw.plot_widgets if c.isVisible()]
    assert len(visible) == PLOT_ROW_CURRENT_DEFAULT * PLOT_COL_CURRENT_DEFAULT == 3


def test_load_populates_variable_list(loaded_window):
    """加载后：变量列表展示全部变量（time/speed/rpm/flag）"""
    mw = loaded_window
    assert mw.loader is not None
    assert mw.loader.datalength == 50

    var_names = set(mw.loader.var_names)
    assert {"time", "speed", "rpm", "flag"} <= var_names
    # 变量列表行数与变量数一致
    assert mw.list_widget.rowCount() == len(mw.loader.var_names)


def test_buttons_enabled_after_load(loaded_window):
    """加载后：分析类按钮全部启用"""
    mw = loaded_window
    for btn in (
        mw.clear_all_plots_btn,
        mw.auto_range_btn,
        mw.auto_y_btn,
        mw.cursor_btn,
        mw.mark_region_btn,
        mw.grid_layout_btn,
    ):
        assert btn.isEnabled(), f"{btn.text()} 应在加载后启用"


def test_drag_variable_to_plot(loaded_window, qapp):
    """拖拽变量到绘图区 = 添加曲线（help.md 3️⃣）"""
    mw = loaded_window
    assert len(mw.plot_widgets) >= 1
    pw = mw.plot_widgets[0].plot_widget

    mime = build_var_mimedata(["speed"])
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.dropEvent(ev)
    qapp.processEvents()

    assert "speed" in pw.curves
    assert len(pw.curves) == 1

    # 再拖一个变量 = 追加（默认添加语义）
    mime2 = build_var_mimedata(["rpm"])
    _keep_alive.append(mime2)
    ev2 = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime2,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.dropEvent(ev2)
    qapp.processEvents()

    assert list(pw.curves) == ["speed", "rpm"]


def test_shift_drag_replaces_curves(loaded_window, qapp):
    """Shift + 拖拽 = 替换（先清空再绘制，help.md 3️⃣）"""
    mw = loaded_window
    pw = mw.plot_widgets[0].plot_widget

    mime = build_var_mimedata(["speed", "rpm"])
    _keep_alive.append(mime)
    ev = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    pw.dropEvent(ev)
    qapp.processEvents()
    assert len(pw.curves) == 2

    mime2 = build_var_mimedata(["flag"])
    _keep_alive.append(mime2)
    ev2 = QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime2,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )
    pw.dropEvent(ev2)
    qapp.processEvents()

    assert list(pw.curves) == ["flag"]


def test_ctrl_r_toggles_cursor(loaded_window, qapp, qtbot):
    """Ctrl+R 快捷键切换游标显示（help.md 5️⃣ / ⌨️）"""
    mw = loaded_window
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata(["speed"])
    _keep_alive.append(mime)
    pw.dropEvent(QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    qapp.processEvents()

    mw.activateWindow()
    qapp.processEvents()
    assert not mw.cursor_btn.isChecked()
    assert mw.cursor_btn.text() == "显示游标"

    qtbot.keyClick(mw, Qt.Key_R, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert mw.cursor_btn.isChecked(), "Ctrl+R 应打开游标"
    assert mw.cursor_btn.text() == "隐藏游标", "按钮文案需随状态切换（术语统一为游标）"

    qtbot.keyClick(mw, Qt.Key_R, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()
    assert not mw.cursor_btn.isChecked(), "再次 Ctrl+R 应关闭游标"
    assert mw.cursor_btn.text() == "显示游标"


def test_ctrl_y_auto_y_after_plot(loaded_window, qapp, qtbot):
    """Ctrl+Y「仅调节 Y 轴」：绘图后触发不崩溃且启用 Y autoRange"""
    mw = loaded_window
    pw = mw.plot_widgets[0].plot_widget
    mime = build_var_mimedata(["speed"])
    _keep_alive.append(mime)
    pw.dropEvent(QDropEvent(
        QPointF(50.0, 50.0),
        Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
        mime, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
    ))
    qapp.processEvents()

    mw.activateWindow()
    qapp.processEvents()
    qtbot.keyClick(mw, Qt.Key_Y, Qt.KeyboardModifier.ControlModifier)
    qapp.processEvents()

    assert bool(pw.view_box.state["autoRange"][1]) is True
