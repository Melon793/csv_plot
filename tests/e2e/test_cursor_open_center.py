"""自由游标打开落点护栏（方案 B + 鼠标快照）。

背景：vline 出生值 x=0 被 setBounds 钳到数据域左端，红线与 Y 轴/曲线
起点重叠，用户以为游标没打开。修复后 toggle_cursor_all(True) 在
setChecked 之后调 _place_free_cursor_on_open：
1. 鼠标在可见子图绘图区内 → 落鼠标 X（打开瞬间一次性快照）；
2. 否则落源 plot 视图中央。钳进游标域后走 sync_crosshair 同步。

覆盖：
- 鼠标在图外：按钮 / Ctrl+R 两条入口同规格落中央；
- 鼠标在图内：落鼠标 X 而非中央（含 Ctrl+R 入口、双图同步）；
- 鼠标在场景内但绘图区外（轴/标题区）→ 回退中央；
- 缩放后落"当前视图"中央而非数据域中央；
- 隐藏再打开重新落点（B 语义：不保留旧位置）；
- 固定游标（1/2 锚定）带 context_x 时立即覆盖落点，不受影响；
- 重开首帧前标签已对齐落点（防"线在新位、标签在旧位"闪烁）。

时序注意：loaded_window 返回时加载链可能尚未收尾
（_is_loading_new_data / _is_interacting 仍为 True），sync_crosshair
会被守卫拦下——真实用户点按钮时加载早已完成，测试须先等收尾。

QCursor.pos 在 offscreen 下读到的是宿主机真实鼠标，不确定——autouse
替身默认钉到屏幕外（强制中央兜底），鼠标在图内的用例自行覆盖。
"""

from __future__ import annotations

import pytest

from PySide6.QtCore import QPoint, QPointF
from PySide6.QtGui import QCursor

from tests.fixtures.waits import settle


@pytest.fixture(autouse=True)
def _mouse_far_away(monkeypatch):
    """默认把全局鼠标钉到屏幕外：落点走视图中央兜底，结果确定。"""
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: QPoint(-10000, -10000)))


def _global_pos_at_plot_fraction(pw, fx, fy=0.5):
    """绘图区场景矩形 (fx, fy) 比例处对应的全局坐标（供 QCursor 替身）。"""
    rect = pw.plot_item.sceneBoundingRect()
    scene_pt = QPoint(
        int(rect.left() + rect.width() * fx),
        int(rect.top() + rect.height() * fy),
    )
    return pw.mapToGlobal(pw.mapFromScene(scene_pt)), scene_pt


def _settle_loading(mw, qtbot):
    qtbot.waitUntil(lambda: not mw._is_loading_new_data, timeout=5000)
    for container in mw.plot_widgets:
        pw = container.plot_widget
        qtbot.waitUntil(lambda pw=pw: not pw._is_interacting, timeout=5000)
    settle()


def _view_center(pw):
    vmin, vmax = pw.view_box.viewRange()[0]
    return (vmin + vmax) / 2


def _open_and_assert_center(mw, pw, qapp):
    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    assert mw.cursor_btn.isChecked()
    assert mw.cursor_mode == "1 free cursor"
    center = _view_center(pw)
    assert pw.vline.value() == pytest.approx(center, abs=1e-6), (
        f"打开后 vline={pw.vline.value()} 应落视图中央 {center}"
    )
    return center


def test_open_cursor_lands_at_view_center_not_domain_edge(loaded_window, qapp, qtbot):
    """按钮入口：打开前钳在数据域左端（旧错觉），打开后落视图中央。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    domain = pw._cursor_x_domain()
    assert domain is not None
    # 前置：出生值 0 被钳到域左端——这正是"看起来没打开"的旧落点
    assert pw.vline.value() == pytest.approx(domain[0], abs=1e-6)

    center = _open_and_assert_center(mw, pw, qapp)
    assert center > domain[0], "视图中央必须离开左边缘，否则护栏失去意义"


def test_ctrl_r_shortcut_centers_cursor(loaded_window, qapp, qtbot):
    """Ctrl+R 入口与按钮同规格。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    mw._on_cursor_shortcut()
    qapp.processEvents()
    assert mw.cursor_btn.isChecked()
    assert pw.vline.value() == pytest.approx(_view_center(pw), abs=1e-6)


def test_open_centers_on_zoomed_view_not_data_domain(loaded_window, qapp, qtbot):
    """缩放到子窗口后打开：落当前视图中央，而非全局数据域中央。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    domain = pw._cursor_x_domain()
    lo = domain[0] + (domain[1] - domain[0]) * 0.2
    hi = domain[0] + (domain[1] - domain[0]) * 0.4
    pw.view_box.setXRange(lo, hi, padding=0)
    qapp.processEvents()
    # setXRange 会置 _is_interacting（sync_crosshair 守卫之一），由定时器清除
    qtbot.waitUntil(lambda: not pw._is_interacting, timeout=5000)
    settle()
    vmin, vmax = pw.view_box.viewRange()[0]
    assert vmin == pytest.approx(lo, abs=1e-6)
    assert vmax == pytest.approx(hi, abs=1e-6)

    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    assert pw.vline.value() == pytest.approx((lo + hi) / 2, abs=1e-6)


def test_reopen_recenters_after_mouse_sync(loaded_window, qapp, qtbot):
    """B 语义：隐藏再打开重新居中，不保留鼠标最后同步的位置。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    center = _open_and_assert_center(mw, pw, qapp)

    domain = pw._cursor_x_domain()
    other = domain[0] + (domain[1] - domain[0]) * 0.8
    mw.cursor_sync_manager.sync_crosshair(other, pw)
    qapp.processEvents()
    assert pw.vline.value() == pytest.approx(other, abs=1e-6)

    mw.cursor_sync_manager.toggle_cursor_all(False)
    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    assert pw.vline.value() == pytest.approx(center, abs=1e-6)


@pytest.mark.full_matrix
def test_open_syncs_center_to_all_visible_plots(two_plots, qapp, qtbot):
    """双图：打开后所有可见子图落同一 x（源 plot 视图中央）。"""
    mw, _page, left, right = two_plots
    _settle_loading(mw, qtbot)
    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    center = _view_center(left)
    assert left.vline.value() == pytest.approx(center, abs=1e-6)
    assert right.vline.value() == pytest.approx(center, abs=1e-6)


def test_anchored_context_x_overrides_center(loaded_window, qapp, qtbot):
    """固定游标不受影响：右键带 context_x，从关闭态直接切锚定仍钉在右键处。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    domain = pw._cursor_x_domain()
    ctx_x = domain[0] + (domain[1] - domain[0]) * 0.3

    # 单锚定：cursor 关着 → set_cursor_mode 内部先 toggle(True)（居中）
    # 再立刻用 context_x 覆盖
    mw.cursor_sync_manager.set_cursor_mode(
        "1 anchored cursor", source_plot=pw, context_x=ctx_x
    )
    qapp.processEvents()
    assert mw.pinned_x_values == [pytest.approx(ctx_x, abs=1e-6)]
    assert pw.vline.value() == pytest.approx(ctx_x, abs=1e-6)

    # 双锚定：第一根仍在 context_x，第二根按视图规则生成
    mw.cursor_sync_manager.toggle_cursor_all(False)
    mw.cursor_sync_manager.set_cursor_mode(
        "2 anchored cursor", source_plot=pw, context_x=ctx_x
    )
    qapp.processEvents()
    assert len(mw.pinned_x_values) == 2
    assert mw.pinned_x_values[0] == pytest.approx(ctx_x, abs=1e-6)
    assert mw.pinned_x_values[1] != pytest.approx(ctx_x, abs=1e-6)


def test_open_with_mouse_inside_plot_lands_at_mouse(loaded_window, qapp, qtbot, monkeypatch):
    """鼠标在绘图区内按打开：落鼠标 X（一次性快照），而非视图中央。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    global_pt, scene_pt = _global_pos_at_plot_fraction(pw, 0.3)
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: global_pt))
    expected_x = pw.plot_item.vb.mapSceneToView(scene_pt).x()
    assert expected_x != pytest.approx(_view_center(pw), abs=0.5), (
        "快照点须明显偏离中央，否则护栏失去判别力"
    )

    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    assert pw.vline.value() == pytest.approx(expected_x, abs=1e-6)


def test_ctrl_r_with_mouse_inside_plot_lands_at_mouse(loaded_window, qapp, qtbot, monkeypatch):
    """Ctrl+R 入口同规格：鼠标在图内 → 落鼠标 X，不等鼠标移动。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    global_pt, scene_pt = _global_pos_at_plot_fraction(pw, 0.7)
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: global_pt))
    expected_x = pw.plot_item.vb.mapSceneToView(scene_pt).x()

    mw._on_cursor_shortcut()
    qapp.processEvents()
    assert mw.cursor_btn.isChecked()
    assert pw.vline.value() == pytest.approx(expected_x, abs=1e-6)


def test_mouse_in_scene_but_outside_plot_area_falls_back_to_center(
    loaded_window, qapp, qtbot, monkeypatch
):
    """鼠标在场景内但绘图区外（标题/轴区）：与 mouse_moved 同口径，回退中央。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    rect = pw.plot_item.sceneBoundingRect()
    # 绘图区正上方 20px：仍在 GraphicsView 场景内，但不在 plot_item 矩形内
    scene_pt = QPoint(int(rect.center().x()), int(rect.top() - 20))
    global_pt = pw.mapToGlobal(pw.mapFromScene(scene_pt))
    assert not pw.plot_item.sceneBoundingRect().contains(scene_pt)
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: global_pt))

    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    assert pw.vline.value() == pytest.approx(_view_center(pw), abs=1e-6)


@pytest.mark.full_matrix
def test_mouse_in_right_plot_syncs_both_plots(two_plots, qapp, qtbot, monkeypatch):
    """双图：鼠标在右图 → 两图都落右图鼠标 X（快照源是鼠标所在图）。"""
    mw, _page, left, right = two_plots
    _settle_loading(mw, qtbot)
    global_pt, scene_pt = _global_pos_at_plot_fraction(right, 0.45)
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: global_pt))
    expected_x = right.plot_item.vb.mapSceneToView(scene_pt).x()
    domain = right._cursor_x_domain()
    # 选点须在数据域内，否则断言的是钳制行为而非快照落点
    assert domain[0] < expected_x < domain[1]

    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    assert left.vline.value() == pytest.approx(expected_x, abs=1e-6)
    assert right.vline.value() == pytest.approx(expected_x, abs=1e-6)


def test_anchored_context_x_wins_over_mouse_snapshot(loaded_window, qapp, qtbot, monkeypatch):
    """鼠标在图内也不影响锚定：context_x（右键处）优先于鼠标快照。"""
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    # 鼠标快照点放 70% 处，右键 context_x 放 30% 处，两者必须分开
    global_pt, _ = _global_pos_at_plot_fraction(pw, 0.7)
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: global_pt))
    domain = pw._cursor_x_domain()
    ctx_x = domain[0] + (domain[1] - domain[0]) * 0.3

    mw.cursor_sync_manager.set_cursor_mode(
        "1 anchored cursor", source_plot=pw, context_x=ctx_x
    )
    qapp.processEvents()
    assert mw.pinned_x_values == [pytest.approx(ctx_x, abs=1e-6)]
    assert pw.vline.value() == pytest.approx(ctx_x, abs=1e-6)


def test_reopen_labels_aligned_before_first_frame(loaded_window, qapp, qtbot, monkeypatch):
    """防"闪一下"：重开当下圆点/数值/X 标签已对齐落点，不依赖 16ms 定时器。

    旧缺陷链：toggle_cursor 把标签画在旧位置 → sync_crosshair 只移线、
    标签刷新委托 16ms 定时器 → 首帧渲染"线在新位、标签在旧位"撕裂态。
    修复：落点后立即 refresh_cursor_labels()，首帧前全部对齐。
    """
    mw = loaded_window
    _settle_loading(mw, qtbot)
    pw = mw.plot_widgets[0].plot_widget
    # 切断 16ms 定时器通路：高负载下它可能合法提前触发并掩盖"无修复"缺陷，
    # no-op 后位置断言只依赖落点后的同步补刷（评审 P2-1 加固）
    monkeypatch.setattr(mw._crosshair_update_timer, "start", lambda *a, **k: None)
    # 标签重建链要求有可见曲线（loaded_window 只加载不绘图）
    assert pw.plot_variable("speed")
    qapp.processEvents()
    qtbot.waitUntil(
        lambda: pw.view_box.sceneBoundingRect().height() > 100, timeout=5000
    )
    _settle_loading(mw, qtbot)
    # 鼠标钉在绘图区 30% 处：重开落点 = 鼠标快照
    global_pt, scene_pt = _global_pos_at_plot_fraction(pw, 0.3)
    monkeypatch.setattr(QCursor, "pos", staticmethod(lambda: global_pt))
    expected_x = pw.plot_item.vb.mapSceneToView(scene_pt).x()

    # 先开一次，再把线同步到别处（模拟"上次关闭前的位置"），然后隐藏
    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    domain = pw._cursor_x_domain()
    old_x = domain[0] + (domain[1] - domain[0]) * 0.8
    assert old_x != pytest.approx(expected_x, abs=0.5)
    mw.cursor_sync_manager.sync_crosshair(old_x, pw)
    qapp.processEvents()
    mw.cursor_sync_manager.toggle_cursor_all(False)
    qapp.processEvents()

    # 重开：不 wait、不 pump，只排空已投递事件。不用 isActive() 断言
    # 定时器未到期——并行负载下重建耗时可超 16ms，定时器合法提前触发；
    # 判别力由下面的位置断言承担（撤掉修复行已验证必红）
    mw.cursor_sync_manager.toggle_cursor_all(True)
    qapp.processEvents()
    assert pw.vline.value() == pytest.approx(expected_x, abs=1e-6)
    # view 坐标 → 场景坐标必须用 mapViewToScene（mapToScene 收的是
    # ViewBox 自身像素局部坐标，混用会差一个 childGroup 变换）
    target_scene_x = pw.plot_item.vb.mapViewToScene(QPointF(expected_x, 0)).x()
    old_scene_x = pw.plot_item.vb.mapViewToScene(QPointF(old_x, 0)).x()
    # 圆点吸附最近采样点（x_actual），与 vline 连续 X 有几个像素偏差；
    # 判据取"到新落点的距离 < 新旧间距的 1/4"，既容吸附又必杀旧位置滞留
    margin = abs(old_scene_x - target_scene_x) / 4
    labels = [it for it in pw.multi_cursor_items if it.isVisible()]
    assert labels, "重开后应有可见的圆点/数值/X 标签"
    checked = 0
    for item in labels:
        if hasattr(item, "points") and item.points():
            # ScatterPlotItem（圆点）：真实位置在散点数据里（view 坐标），
            # 经 childGroup 变换 mapToScene 才是场景位置
            sx = item.mapToScene(item.points()[0].pos()).x()
        elif hasattr(item, "toPlainText"):
            # TextItem（数值/X 标签）：scenePos 即定位点
            sx = item.scenePos().x()
        else:
            # 引导线 PlotCurveItem 的 scenePos 是局部原点而非线段位置，
            # 无判别意义，跳过
            continue
        checked += 1
        assert abs(sx - target_scene_x) < margin, (
            f"标签 {item} 仍滞留在旧位置附近 x={sx}，"
            f"应对齐落点场景 x={target_scene_x}（旧位置 {old_scene_x}）"
        )
    assert checked >= 2, "至少应校验到圆点与一个文本标签"

