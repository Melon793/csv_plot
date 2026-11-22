"""C2：显示名/换算共享入口 + legend 渲染 + 搬运/暂存保真 + 播报 parity。"""

import numpy as np

from src.ui.drag_drop import CurveStyleSnapshot
from src.ui.widgets.curve_shelf import ShelfEntry


def _x():
    return np.arange(3, dtype=float)


def _y():
    return np.array([1.0, 2.0, 3.0])


def _add(pw, name="a"):
    assert pw.add_variable_to_plot(name, x_values=_x(), y_values=_y())
    return pw.curves[name]


def _status_spy(pw):
    """宿主状态栏记录器（window() 返回 FakeHost）。"""
    calls = []
    host = pw.window()
    host._show_status_message = lambda msg, *a, **kw: calls.append(msg)
    return calls


class TestDisplayNameEntry:
    def test_rename_updates_legend_keeps_anchor(self, plot_factory):
        pw = plot_factory()
        _add(pw)
        mcm = pw._multi_curve_manager
        assert mcm.apply_curve_display_name("a", "扭矩A") is True
        html = mcm._build_legend_html()
        assert "扭矩A" in html
        assert "curve:///a" in html  # 锚点仍用内部名
        assert ">a</a>" not in html  # 整串替换，不追加单位

    def test_empty_resets_to_default(self, plot_factory):
        pw = plot_factory()
        _add(pw)
        mcm = pw._multi_curve_manager
        mcm.apply_curve_display_name("a", "扭矩A")
        assert mcm.apply_curve_display_name("a", "") is True
        assert ">a</a>" in mcm._build_legend_html()

    def test_same_value_noop(self, plot_factory):
        pw = plot_factory()
        _add(pw)
        mcm = pw._multi_curve_manager
        mcm.apply_curve_display_name("a", "扭矩A")
        assert mcm.apply_curve_display_name("a", "扭矩A") is False

    def test_missing_var_returns_false(self, plot_factory):
        pw = plot_factory()
        assert pw._multi_curve_manager.apply_curve_display_name("nope", "x") is False

    def test_invalid_ref_rename(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        pw.demote_to_invalid("a", ci.color, ci.visible)
        mcm = pw._multi_curve_manager
        assert mcm.apply_curve_display_name("a", "扭矩A") is True
        assert pw.invalid_curves["a"].display_name == "扭矩A"
        html = mcm._build_legend_html()
        assert "扭矩A" in html and "（缺失）" in html

    def test_broadcast_wording(self, plot_factory):
        pw = plot_factory()
        _add(pw)
        calls = _status_spy(pw)
        mcm = pw._multi_curve_manager
        mcm.apply_curve_display_name("a", "扭矩A")
        mcm.apply_curve_display_name("a", "")
        assert any("扭矩A" in m for m in calls)
        assert any("重置" in m for m in calls)


class TestTransformEntry:
    def test_projects_curve_and_emits(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        emitted = []
        pw.curves_changed.connect(lambda: emitted.append(1))
        mcm = pw._multi_curve_manager
        assert mcm.apply_curve_transform("a", 2.0, -10.0) is True
        np.testing.assert_allclose(ci.y_data, [-8.0, -6.0, -4.0])
        np.testing.assert_allclose(ci.curve.getData()[1], [-8.0, -6.0, -4.0])
        np.testing.assert_allclose(ci.y_raw, _y())
        assert emitted

    def test_reapply_from_raw_not_cumulative(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        mcm = pw._multi_curve_manager
        mcm.apply_curve_transform("a", 2.0, 0.0)
        mcm.apply_curve_transform("a", 3.0, 0.0)
        np.testing.assert_allclose(ci.y_data, [3.0, 6.0, 9.0])

    def test_enum_rejected(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.y_format = "enum"
        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0) is False
        np.testing.assert_allclose(ci.y_data, _y())

    def test_s_time_rejected(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.y_format = "s"
        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0) is False

    def test_non_finite_rejected(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        mcm = pw._multi_curve_manager
        assert mcm.apply_curve_transform("a", float("nan"), 0.0) is False
        assert mcm.apply_curve_transform("a", 1.0, float("inf")) is False
        np.testing.assert_allclose(ci.y_data, _y())

    def test_invalid_var_rejected(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        pw.demote_to_invalid("a", ci.color, ci.visible)
        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0) is False

    def test_same_value_noop(self, plot_factory):
        pw = plot_factory()
        _add(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 1.0, 0.0) is False

    def test_broadcast_formula(self, plot_factory):
        pw = plot_factory()
        _add(pw)
        calls = _status_spy(pw)
        pw._multi_curve_manager.apply_curve_transform("a", 2.0, -10.0)
        assert any("2·y − 10" in m for m in calls)


class TestTransferFidelity:
    def test_snapshot_carries_settings(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.display_name, ci.factor, ci.offset = "扭矩A", 2.0, -10.0
        snap = CurveStyleSnapshot.from_curve_info(ci)
        assert snap.display_name == "扭矩A"
        assert snap.factor == 2.0 and snap.offset == -10.0

    def test_shelf_entry_carries_settings(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.display_name, ci.factor, ci.offset = "扭矩A", 2.0, -10.0
        entry = ShelfEntry.from_curve_info("a", ci)
        assert entry.display_name == "扭矩A"
        assert entry.factor == 2.0 and entry.offset == -10.0

    def test_apply_transfer_style_lands_settings(self, plot_factory):
        pw = plot_factory()
        src = _add(pw, "a")
        src.display_name, src.factor, src.offset = "扭矩A", 2.0, -10.0
        snap = CurveStyleSnapshot.from_curve_info(src)
        _add(pw, "b")
        pw._apply_transfer_style("b", snap)
        dst = pw.curves["b"]
        assert dst.display_name == "扭矩A"
        assert dst.factor == 2.0 and dst.offset == -10.0
        np.testing.assert_allclose(dst.y_data, [-8.0, -6.0, -4.0])

    def test_shelf_entry_lands_settings(self, plot_factory):
        pw = plot_factory()
        src = _add(pw, "a")
        src.display_name, src.factor, src.offset = "扭矩A", 2.0, -10.0
        entry = ShelfEntry.from_curve_info("a", src)
        _add(pw, "b")
        pw._apply_transfer_style("b", entry)
        dst = pw.curves["b"]
        assert dst.display_name == "扭矩A"
        np.testing.assert_allclose(dst.y_data, [-8.0, -6.0, -4.0])

    def test_default_snapshot_noop(self, plot_factory):
        pw = plot_factory()
        _add(pw, "a")
        snap = CurveStyleSnapshot.from_curve_info(pw.curves["a"])
        _add(pw, "b")
        pw._apply_transfer_style("b", snap)
        dst = pw.curves["b"]
        assert dst.display_name == "" and dst.factor == 1.0 and dst.offset == 0.0
        np.testing.assert_allclose(dst.y_data, _y())