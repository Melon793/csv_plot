"""多列紧凑宽度 / 跨组向外分裂 / 滞回分状态护栏（offscreen）。

批次3 契约：
- 单游标列方向左右交替（右/左/右…），某侧无预算时退化单侧；
- 列 x 按同侧累计偏移（宽列后接窄列不自重叠，左侧按右缘镜像锚定），
  列宽取列内实际最大宽；
- 双游标跨组碰撞 → 组0 左/组1 右向外分裂，先校验空间预算，
  通过且零重叠 + 全部在视口内才接受，否则回退原布局；
- 列数滞回按 cursor_id 分状态，组间交替喂入不再互相饿死。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from PySide6.QtCore import QPointF

from src.ui.widgets.cursor_manager import CursorManager
from tests.component.test_cursor_label_edge import (
    _refresh,
    _setup,
    _visible_labels,
)


def _reset_view(pw, x0, x1, y0, y1):
    """放宽 Y limits 后设置视窗（绘图期 limits 会钳制 setYRange）。"""
    pw.plot_item.setLimits(yMin=None, yMax=None)
    vb = pw.view_box
    vb.setXRange(x0, x1, padding=0.05)
    vb.setYRange(y0, y1, padding=0)


def _text_width(pw, lbl) -> float:
    fm = pw._cursor_manager._cached_font_metrics
    return fm.horizontalAdvance(lbl.toPlainText()) + 12


def _label_rects(pw):
    h = pw._cursor_manager._cached_label_height_pixels
    rects = []
    for lbl in _visible_labels(pw):
        pos = lbl.scenePos()
        rects.append((pos.x(), pos.y(), _text_width(pw, lbl), h))
    return rects


def _overlap_pairs(rects) -> int:
    hits = 0
    for i in range(len(rects)):
        ax, ay, aw, ah = rects[i]
        for j in range(i + 1, len(rects)):
            bx, by, bw, bh = rects[j]
            if ax < bx + bw and bx < ax + aw and abs(ay - by) < max(ah, bh):
                hits += 1
    return hits


def _column_xs(pw, labels) -> list[float]:
    xs = []
    for lbl in labels:
        x = round(lbl.scenePos().x(), 3)
        if not any(abs(x - v) < 0.5 for v in xs):
            xs.append(x)
    return sorted(xs)


def _dual_columns_df():
    return pd.DataFrame(
        {f"y{i}": np.full(100, 98.0 + i) for i in range(5)}
    )


# ---------- ① 紧凑列宽：同侧宽列后接窄列不重叠 ----------

def test_compact_columns_wide_then_narrow_no_overlap(
    plot_factory, qapp, monkeypatch
):
    # 高 y（先分配）= 宽文本，低 y = 窄文本 → 第1列宽、第2列窄；
    # 游标贴左缘 → 左侧无预算，两列都退化到右侧（单侧累计口径）
    df = pd.DataFrame({
        "a": np.full(100, 1234.5),
        "b": np.full(100, 2345.6),
        "c": np.full(100, 3456.7),
        "d": np.full(100, 1.0),
        "e": np.full(100, 2.0),
        "f": np.full(100, 3.0),
    })
    pw = _setup(plot_factory, df, size=(700, 160))
    # 显式固定两列：列宽逻辑与字体度量/容量无关（VM 缺字体时高度容量
    # 会自然收敛到 1 列，掩盖累计偏移断言）。
    monkeypatch.setattr(
        pw._cursor_manager, "_apply_column_hysteresis",
        lambda new_count, cursor_id=0: 2,
    )
    # 显式固定容量，避免字体/DPI 影响列内容（填满分配下会改变哪列宽）
    monkeypatch.setattr(
        pw._cursor_manager, "_column_capacity",
        lambda available, h, gap: 4,
    )
    _reset_view(pw, 0, 99, 0, 200000)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(2.0)  # 左侧预算≈0 → 交替退化为全右
    _refresh(pw, times=4)

    mgr = pw._cursor_manager
    labels = _visible_labels(pw)
    assert len(labels) == 6
    xs = _column_xs(pw, labels)
    assert len(xs) == 2, f"应形成两列，实际列 x={xs}"

    cursor_scene_x = pw.view_box.mapViewToScene(QPointF(2.0, 100.0)).x()
    assert all(x > cursor_scene_x for x in xs), (
        f"游标贴左缘时两列都应在右侧，实际列 x={xs}"
    )

    col_inner = [l for l in labels if abs(l.scenePos().x() - xs[0]) < 0.5]
    col_outer = [l for l in labels if abs(l.scenePos().x() - xs[1]) < 0.5]
    w_inner = max(_text_width(pw, l) for l in col_inner)
    assert w_inner > max(_text_width(pw, l) for l in col_outer), (
        "前置：第1列（宽文本）应比第2列（窄文本）宽"
    )
    gap = mgr.LABEL_COL_GAP
    assert abs(xs[0] - (cursor_scene_x + mgr.LABEL_GAP_TO_CURSOR)) < 0.5
    assert abs(
        xs[1] - (cursor_scene_x + mgr.LABEL_GAP_TO_CURSOR + w_inner + gap)
    ) < 0.5, "第2列应按第1列实际宽累计（宽列后接窄列不自重叠）"
    assert xs[1] >= xs[0] + w_inner + gap - 0.5, (
        "第2列压到了第1列标签上（未按累计偏移）"
    )


# ---------- ② 交替两列：宽标签在左列，左列右缘贴 GAP ----------

def test_wide_labels_in_left_column_anchored_to_gap(
    plot_factory, qapp, monkeypatch
):
    # 高 y = 窄文本 → 右侧最内列；低 y = 宽文本 → 左侧最内列
    df = pd.DataFrame({
        "a": np.full(100, 3.0),
        "b": np.full(100, 2.0),
        "c": np.full(100, 1.0),
        "d": np.full(100, 0.1234567),
        "e": np.full(100, 0.2345678),
        "f": np.full(100, 0.3456789),
    })
    pw = _setup(plot_factory, df, size=(700, 160))
    monkeypatch.setattr(
        pw._cursor_manager, "_apply_column_hysteresis",
        lambda new_count, cursor_id=0: 2,
    )
    # 容量 3：6 个 → 3+3；交替下第 2 列在左侧，宽标签（低 y）落在左列
    monkeypatch.setattr(
        pw._cursor_manager, "_column_capacity",
        lambda available, h, gap: 3,
    )
    _reset_view(pw, 0, 99, 0, 200000)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw, times=4)

    mgr = pw._cursor_manager
    labels = _visible_labels(pw)
    assert len(labels) == 6
    cursor_scene_x = pw.view_box.mapViewToScene(QPointF(50.0, 100.0)).x()
    right_col = [l for l in labels if l.scenePos().x() > cursor_scene_x]
    left_col = [l for l in labels if l.scenePos().x() < cursor_scene_x]
    assert len(right_col) == 3 and len(left_col) == 3, (
        f"交替应形成 3 右 + 3 左，实际 {len(right_col)} 右 + {len(left_col)} 左"
    )

    right_xs = {round(l.scenePos().x(), 3) for l in right_col}
    left_xs = {round(l.scenePos().x(), 3) for l in left_col}
    assert len(right_xs) == 1 and len(left_xs) == 1

    w_right = max(_text_width(pw, l) for l in right_col)
    w_left = max(_text_width(pw, l) for l in left_col)
    assert w_left > w_right, "前置：左侧列应比右侧列宽"

    right_x = right_xs.pop()
    left_x = left_xs.pop()
    # 右侧按左缘锚定，左侧按右缘锚定（镜像累计）
    assert abs(
        right_x - (cursor_scene_x + mgr.LABEL_GAP_TO_CURSOR)
    ) < 0.5
    assert abs(
        (left_x + w_left) - (cursor_scene_x - mgr.LABEL_GAP_TO_CURSOR)
    ) < 0.5, "左列右缘应贴光标−GAP（按自身实际宽镜像累计）"


# ---------- ③ 单列退化：与既有行为一致 ----------

def test_single_column_degenerates_to_one_offset(plot_factory, qapp):
    df = pd.DataFrame(
        {f"y{i}": np.full(100, 100.0 + 0.1 * i) for i in range(5)}
    )
    pw = _setup(plot_factory, df, size=(500, 500))
    _reset_view(pw, 0, 99, 95, 105)
    qapp.processEvents()
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw, times=4)

    labels = _visible_labels(pw)
    assert len(labels) == 5
    xs = _column_xs(pw, labels)
    assert len(xs) == 1, f"单列场景不应分列：{xs}"
    cursor_scene_x = pw.view_box.mapViewToScene(QPointF(50.0, 100.0)).x()
    assert abs(xs[0] - (cursor_scene_x + 10.0)) < 0.5


# ---------- ④ 跨组向外分裂：重叠 0 且全部在视口内 ----------

def test_cross_group_outward_split_zero_overlap(plot_factory, qapp):
    pw = _setup(plot_factory, _dual_columns_df(), size=(500, 420))
    _reset_view(pw, 0, 99, 80, 120)
    qapp.processEvents()
    pw.plot_context.cursor_mode = "2 anchored cursor"
    pw._cursor_manager.pinned_x_values = [48.0, 52.0]
    pw.vline.setVisible(True)
    pw.vline2.setVisible(True)
    _refresh(pw, times=4)

    labels = _visible_labels(pw)
    assert len(labels) == 10
    rects = _label_rects(pw)
    assert _overlap_pairs(rects) == 0, "向外分裂后仍存在跨组重叠"

    view_rect = pw.view_box.sceneBoundingRect()
    for x, y, w, h in rects:
        assert view_rect.left() <= x and x + w <= view_rect.right() + 0.5
        assert view_rect.top() <= y - h / 2 and y + h / 2 <= view_rect.bottom()

    # 组0 标签在左游标左侧、组1 在右游标右侧（确实向外分裂）
    c0 = pw.view_box.mapViewToScene(QPointF(48.0, 100.0)).x()
    c1 = pw.view_box.mapViewToScene(QPointF(52.0, 100.0)).x()
    assert min(x for x, *_ in rects) < c0
    assert max(x for x, *_ in rects) > c1


# ---------- ⑤ 空间不足：回退原布局并接受重叠 ----------

def test_cross_group_avoid_falls_back_when_space_insufficient(plot_factory, qapp):
    pw = _setup(plot_factory, _dual_columns_df(), size=(500, 420))
    _reset_view(pw, 0, 99, 80, 120)
    qapp.processEvents()
    pw.plot_context.cursor_mode = "2 anchored cursor"
    pw._cursor_manager.pinned_x_values = [6.0, 10.0]  # 组0 左侧空间不足
    pw.vline.setVisible(True)
    pw.vline2.setVisible(True)
    _refresh(pw, times=4)

    labels = _visible_labels(pw)
    assert len(labels) == 10
    view_rect = pw.view_box.sceneBoundingRect()
    for lbl in labels:
        assert view_rect.left() <= lbl.scenePos().x() <= view_rect.right()

    # 回退：组0 仍在光标右侧（未接受强制左布局），且如实保留重叠
    c0 = pw.view_box.mapViewToScene(QPointF(6.0, 100.0)).x()
    assert all(lbl.scenePos().x() > c0 for lbl in labels[:5])
    assert _overlap_pairs(_label_rects(pw)) > 0, "空间不足时应回退并接受重叠"


# ---------- ⑥ 列数滞回按 cursor_id 分状态 + 键清理 ----------

def test_column_hysteresis_per_cursor_and_key_cleanup(plot_factory, qapp):
    pw = _setup(plot_factory, _dual_columns_df(), size=(500, 420))
    _reset_view(pw, 0, 99, 80, 120)
    qapp.processEvents()
    mgr = pw._cursor_manager

    # 单游标旧行为：初始 1 列，连续 3 帧才切到 2
    assert mgr._apply_column_hysteresis(2, 0) == 1
    assert mgr._apply_column_hysteresis(2, 0) == 1
    assert mgr._apply_column_hysteresis(2, 0) == 2

    # 双游标交替 10 帧：两组各自收敛，互不喂入对方计数器
    mgr._column_state.clear()
    for _ in range(10):
        mgr._apply_column_hysteresis(2, 0)
        mgr._apply_column_hysteresis(3, 1)
    assert mgr._column_state[0][0] == 2
    assert mgr._column_state[1][0] == 3

    # 实际双游标渲染：两组都有状态键
    mgr._column_state.clear()
    pw.plot_context.cursor_mode = "2 anchored cursor"
    mgr.pinned_x_values = [48.0, 52.0]
    pw.vline.setVisible(True)
    pw.vline2.setVisible(True)
    _refresh(pw, times=4)
    assert set(mgr._column_state) == {0, 1}

    # 回到单游标：已消失的 cursor 1 状态键被回收
    pw.plot_context.cursor_mode = "1 anchored cursor"
    mgr.pinned_x_values = []
    mgr.pinned_x_value = 50.0
    pw.vline.setVisible(True)
    _refresh(pw, times=4)
    assert 1 not in mgr._column_state
    assert 0 in mgr._column_state