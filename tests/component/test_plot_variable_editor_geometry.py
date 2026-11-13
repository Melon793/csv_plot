"""绘图变量编辑器：窗口几何记忆与列宽自适应组件测试。

覆盖（需求 2 方案 B）：
- 首开自适应：初始宽 = 固定列宽和 + 变量名最小宽 + 余量，clamp 到
  主窗口宽 × 0.9；高度静态 600；居中主窗口
- closeEvent 保存几何 → 新编辑器恢复（位置 + 大小）
- 失效屏幕守卫：记忆位置落已拔掉的屏幕（不与任何现存屏幕相交）→
  丢弃记忆回退初始几何
- 损坏几何（restoreGeometry False）→ 回退初始几何
- 首帧数据填充后固定列内容自适应列宽（下限防截断）
"""

import types

import pandas as pd
import pytest
from PySide6.QtCore import QByteArray, QSize

from src.core.settings import ConfigKey
from src.ui.plot_variable_editor import PlotVariableEditorDialog

NAMES = ["a", "b"]


@pytest.fixture()
def editor_host(plot_factory, qapp, app_settings):
    """带数据曲线与编辑器的隔离环境（settings 已重定向 + 几何键清零）。"""
    # ini 残留跨用例污染（fixture 只重置单例不删 ini），先显式清键
    app_settings.set_value(ConfigKey.PLOT_VARIABLE_EDITOR_GEOMETRY, "")
    df = pd.DataFrame({name: [float(i), float(i + 1)] for i, name in enumerate(NAMES)})
    pw = plot_factory(df)
    host = pw.window()
    host.loader = types.SimpleNamespace(var_names=list(NAMES), units={})
    host.data_validity = {}
    for name in NAMES:
        assert pw.add_variable_to_plot(name)
    dlg = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    yield dlg, pw, host
    if dlg.isVisible():
        dlg.close()
    dlg.deleteLater()


def _expected_initial_width(dlg, host):
    fixed = sum(dlg.var_table.columnWidth(c) for c in (0, 2, 3, 4))
    return min(fixed + 260 + 24 + 40, int(host.width() * 0.9))


def test_first_open_adaptive_geometry(editor_host):
    """无记忆首开：初始宽按列宽计算 + clamp 主窗宽×0.9，高度 600，居中主窗口。"""
    dlg, _pw, host = editor_host
    assert dlg.height() == 600
    assert dlg.width() == _expected_initial_width(dlg, host)
    # 居中主窗口（整除取整允许 ±1）
    assert abs(dlg.x() - (host.x() + (host.width() - dlg.width()) // 2)) <= 1
    assert abs(dlg.y() - (host.y() + (host.height() - dlg.height()) // 2)) <= 1


def test_geometry_saved_on_close_and_restored(editor_host, qapp):
    """closeEvent 保存几何；新开编辑器恢复上次位置与大小。

    位置允许 ±4px 容差：saveGeometry 存 frame 几何，offscreen 平台
    Tool 窗边框宽度与恢复后 client 原点存在 1-2px 偏移。
    """
    dlg, pw, _host = editor_host
    dlg.show()
    dlg.resize(700, 500)
    dlg.move(60, 70)
    dlg.close()  # 触发 closeEvent 保存
    qapp.processEvents()

    dlg2 = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    try:
        assert dlg2.size() == QSize(700, 500)
        assert abs(dlg2.x() - 60) <= 4
        assert abs(dlg2.y() - 70) <= 4
    finally:
        dlg2.deleteLater()


def test_stale_screen_geometry_falls_back(editor_host, qapp):
    """记忆位置在所有现存屏幕之外（多显示器拔插）：回退初始几何。"""
    dlg, pw, _host = editor_host
    dlg.show()
    dlg.move(-9999, -9999)  # 不与任何现存屏幕相交
    dlg.close()
    qapp.processEvents()

    dlg2 = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    try:
        assert dlg2.x() > -9000, "失效屏幕记忆应被丢弃"
        assert dlg2.size() == QSize(
            _expected_initial_width(dlg2, pw.window()), 600
        )
    finally:
        dlg2.deleteLater()


def test_corrupt_geometry_falls_back(editor_host, app_settings, qapp):
    """几何数据损坏（restoreGeometry False）：回退初始几何。"""
    dlg, pw, _host = editor_host
    app_settings.set_plot_variable_editor_geometry(
        QByteArray(b"garbage-not-a-geometry-blob")
    )

    dlg2 = PlotVariableEditorDialog(pw)
    qapp.processEvents()
    try:
        assert dlg2.size() == QSize(
            _expected_initial_width(dlg2, pw.window()), 600
        )
    finally:
        dlg2.deleteLater()


def test_minimum_size_enforced(editor_host):
    """最小尺寸约束：用户无法拖到不可用的小窗。"""
    dlg, _pw, _host = editor_host
    assert dlg.minimumWidth() == 560
    assert dlg.minimumHeight() == 400


def test_fixed_columns_content_sized(editor_host):
    """首帧数据填充后固定列内容自适应：宽度 >= 下限（防极端窄值截断）。"""
    dlg, _pw, _host = editor_host
    minimums = {0: 44, 2: 64, 3: 72, 4: 76}
    for col, minimum in minimums.items():
        width = dlg.var_table.columnWidth(col)
        assert width >= minimum, f"列 {col} 宽度 {width} 低于下限 {minimum}"
    assert dlg._columns_sized is True
