"""短行列容量、填充分配与垂直间距（offscreen）。

现场现象：6 个标签在“刚好放下”的行高下被拆成 3+3，左右拉动还会
跳列。护栏：
1. 容量精确口径下，行高足够时 6 个标签稳定单列；
2. 容量为 5 时分配为 5+1（不是 3+3）；
3. 相邻标签中心距 = 布局高 + LABEL_VERTICAL_GAP（2px），
   文本盒可见间隙 = 间距 + 2px 补偿。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.component.test_cursor_label_edge import (
    _refresh,
    _setup,
    _visible_labels,
)


def _df6() -> pd.DataFrame:
    x = np.arange(100.0)
    vals = (1015.1, 1015.1, 1014.8, 1013.3, 89.81, 0.72)
    return pd.DataFrame({f"y{i}": np.full(100, v) for i, v in enumerate(vals)})


def _df2_cluster() -> pd.DataFrame:
    x = np.arange(100.0)
    return pd.DataFrame({"a": x * 2.0, "b": x * 2.0 + 0.05})


def _wide_y(vb) -> None:
    vb.setLimits(yMin=-1e9, yMax=1e9)
    vb.disableAutoRange()
    vb.setYRange(0, 200, padding=0)


def _capacity(pw) -> int:
    mgr = pw._cursor_manager
    h = mgr._cached_label_height_pixels
    vb = pw.view_box
    available = vb.height() - 2 * h * 0.1
    return mgr._column_capacity(available, h, mgr.LABEL_VERTICAL_GAP)


def _resize_to_capacity(pw, qapp, target: int, max_h: int = 420) -> int:
    """逐档拉高窗口，直到单列容量 == target；返回实际窗口高。"""
    for hgt in range(150, max_h + 1, 2):
        pw.resize(700, hgt)
        qapp.processEvents()
        if _capacity(pw) >= target:
            return hgt
    raise AssertionError(f"未找到容量 {target} 的窗口高")


def _column_xs(labels) -> list[float]:
    xs = []
    for lbl in labels:
        x = round(lbl.scenePos().x(), 3)
        if not any(abs(x - v) < 0.5 for v in xs):
            xs.append(x)
    return sorted(xs)


def _seed(pw) -> None:
    pw.vline.setVisible(True)
    pw.vline.setPos(50.0)
    _refresh(pw)  # 建立字体/标签高度缓存


def test_six_labels_stay_single_column_at_exact_fit(plot_factory, qapp):
    pw = _setup(plot_factory, _df6(), size=(700, 200))
    _wide_y(pw.view_box)
    _seed(pw)

    _resize_to_capacity(pw, qapp, 6)
    _refresh(pw, times=4)  # 越过列数滞回

    labels = _visible_labels(pw)
    assert len(labels) == 6
    xs = _column_xs(labels)
    assert len(xs) == 1, f"容量足够 6 个时应单列，实际列 x={xs}"
    # 相邻中心距 = 布局高 + 间距（gap 由常量决定，不再按 27 硬算）
    mgr = pw._cursor_manager
    step = mgr._cached_label_height_pixels + mgr.LABEL_VERTICAL_GAP
    ys = sorted(lbl.scenePos().y() for lbl in labels)
    for a, b in zip(ys, ys[1:]):
        assert b - a == pytest.approx(step, abs=0.5), "单列堆叠步距不符"

    # 复现现场：左右移动游标（连续刷新）不得翻回两列
    for x in (45.0, 52.0, 48.0, 55.0):
        pw.vline.setPos(x)
        _refresh(pw)
        assert len(_column_xs(_visible_labels(pw))) == 1, (
            f"移动游标到 x={x} 后翻回多列"
        )


def test_capacity_five_distributes_five_plus_one(plot_factory, qapp):
    pw = _setup(plot_factory, _df6(), size=(700, 200))
    _wide_y(pw.view_box)
    _seed(pw)

    _resize_to_capacity(pw, qapp, 5)
    _refresh(pw, times=4)

    labels = _visible_labels(pw)
    assert len(labels) == 6
    xs = _column_xs(labels)
    assert len(xs) == 2, f"容量 5 时 6 个标签应两列，实际列 x={xs}"
    col0 = [lbl for lbl in labels if abs(lbl.scenePos().x() - xs[0]) < 0.5]
    col1 = [lbl for lbl in labels if abs(lbl.scenePos().x() - xs[1]) < 0.5]
    assert len(col0) == 5 and len(col1) == 1, (
        f"填满容量分配应为 5+1，实际 {len(col0)}+{len(col1)}"
    )


def test_vertical_gap_visual_spacing(plot_factory, qapp):
    pw = _setup(plot_factory, _df2_cluster(), size=(700, 420))
    _wide_y(pw.view_box)
    _seed(pw)
    _refresh(pw, times=3)

    labels = sorted(_visible_labels(pw), key=lambda lbl: lbl.scenePos().y())
    assert len(labels) == 2
    mgr = pw._cursor_manager
    center_gap = labels[1].scenePos().y() - labels[0].scenePos().y()
    assert center_gap == pytest.approx(
        mgr._cached_label_height_pixels + mgr.LABEL_VERTICAL_GAP, abs=0.5
    )
    box_gap = center_gap - labels[0].boundingRect().height()
    assert box_gap == pytest.approx(mgr.LABEL_VERTICAL_GAP + 2, abs=1.0), (
        f"文本框可见间隙={box_gap}px，预期 {mgr.LABEL_VERTICAL_GAP + 2}px"
    )