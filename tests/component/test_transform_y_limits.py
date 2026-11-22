"""X3（F4/D15）：换算后 Y limits/视野重算链。

真机缺陷：改 factor/offset 后右键自动 Y 仍落在旧范围——limits 在加曲线时
按原始 y 设定，变换只 setData 不重算，autoRange 被陈旧 limits 钳制。
"""

import numpy as np
import pytest

from src.core.data_types import InvalidCurveRef
from tests.fixtures.waits import settle


def _limits(pw):
    return pw.view_box.state["limits"]["yLimits"]


def _view_y(pw):
    return tuple(pw.view_box.viewRange()[1])


def _add_raw(pw, name="a", y=(1.0, 2.0, 3.0)):
    x = np.arange(len(y), dtype=float)
    assert pw.add_variable_to_plot(name, x_values=x, y_values=np.asarray(y, float))
    return pw.curves[name]


class TestTransformRecalc:
    def test_limits_cover_projected_range(self, plot_factory):
        pw = plot_factory()
        _add_raw(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 10.0)
        lo, hi = _limits(pw)
        assert lo <= 12.0 and hi >= 16.0, "limits 未覆盖投影后的 [12,16]"

    def test_view_kept_when_inside_new_limits(self, plot_factory):
        """D15：视野在新 limits 内则原样保留（连续调参不跳视图）。"""
        pw = plot_factory()
        _add_raw(pw)
        before = _view_y(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 0.0)
        assert _view_y(pw) == pytest.approx(before)
        lo, hi = _limits(pw)
        assert lo <= 2.0 and hi >= 6.0

    def test_shrinking_transform_clamps_view(self, plot_factory):
        """D15：视野超出新 limits 时由 pyqtgraph 夹紧到新 limits。"""
        pw = plot_factory()
        _add_raw(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 0.001, 0.0)
        lo, hi = _limits(pw)
        view_lo, view_hi = _view_y(pw)
        assert lo <= 0.001 and hi >= 0.003
        assert view_lo >= lo - 1e-6
        assert view_hi <= hi + 1e-6
        assert view_hi < 1.0, "视野应被收缩后的 limits 夹紧"

    def test_auto_y_after_transform_covers_projected(self, plot_factory):
        """真机路径：换算后右键「自动 Y」必须落在投影范围而非旧 raw 范围。

        offscreen 无 paint 事件，queueUpdateAutoRange 不会自行落地，
        显式 updateAutoRange() 复现画布刷新时的那次重算。
        """
        pw = plot_factory()
        _add_raw(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 10.0)
        pw.auto_y_in_x_range()
        pw.view_box.updateAutoRange()
        settle(3)
        lo, hi = _view_y(pw)
        assert lo <= 12.5, f"自动 Y 下限 {lo} 未覆盖投影值 12"
        assert hi >= 15.5, f"自动 Y 上限 {hi} 未覆盖投影值 16"


class TestReviveWithTransform:
    def test_pdm_first_curve_path_uses_projected_limits(self, plot_factory):
        """空图复活带换算：首条路径曾用 raw y 定 limits（真 stale）。"""
        pw = plot_factory()
        pw.invalid_curves["a"] = InvalidCurveRef(
            var_name="a", color="red", visible_preference=True,
            factor=2.0, offset=10.0,
        )
        assert pw.plot_variable("a")
        ci = pw.curves["a"]
        np.testing.assert_allclose(ci.y_data, [12.0, 14.0, 16.0])
        lo, hi = _limits(pw)
        assert lo <= 12.0 and hi >= 16.0

    def test_add_path_revive_uses_projected_limits(self, plot_factory):
        pw = plot_factory()
        _add_raw(pw, "a", (1.0, 2.0, 3.0))
        pw.invalid_curves["b"] = InvalidCurveRef(
            var_name="b", color="red", visible_preference=True,
            factor=2.0, offset=10.0,
        )
        assert pw.add_variable_to_plot("b", x_values=np.arange(3.0),
                                       y_values=np.array([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(pw.curves["b"].y_data, [18.0, 20.0, 22.0])
        lo, hi = _limits(pw)
        assert lo <= 18.0 and hi >= 22.0


class TestRecreateAndDemote:
    def test_hidden_transformed_recreate_keeps_limits_of_visible(self, plot_factory):
        """评审 P1-2：重建瞬间 add 置 visible=True，重算必须在显隐恢复之后。"""
        pw = plot_factory()
        _add_raw(pw, "a", (1.0, 2.0, 3.0))
        _add_raw(pw, "b", (40.0, 50.0, 60.0))
        assert pw._multi_curve_manager.apply_curve_transform("a", 100.0, 0.0)
        assert pw._multi_curve_manager.apply_curve_visibility("a", False)
        assert _limits(pw) == pytest.approx([30.0, 70.0])

        pw._multi_curve_manager._recreate_curve("a")

        assert pw.curves["a"].visible is False
        assert _limits(pw) == pytest.approx([30.0, 70.0]), "隐藏曲线污染了 limits"

    def test_recreate_keeps_projected_limits(self, plot_factory):
        pw = plot_factory()
        _add_raw(pw)
        assert pw._multi_curve_manager.apply_curve_transform("a", 2.0, 10.0)
        pw._multi_curve_manager._recreate_curve("a")
        ci = pw.curves["a"]
        np.testing.assert_allclose(ci.y_data, [12.0, 14.0, 16.0])
        lo, hi = _limits(pw)
        assert lo <= 12.0 and hi >= 16.0

    def test_demote_shrinks_limits_to_remaining(self, plot_factory):
        pw = plot_factory()
        _add_raw(pw, "a", (1.0, 2.0, 3.0))
        _add_raw(pw, "b", (40.0, 50.0, 60.0))
        assert _limits(pw)[1] >= 60.0

        pw.demote_to_invalid("b", "gray", True)

        lo, hi = _limits(pw)
        assert hi == pytest.approx(4.0, abs=1e-9), "删掉大值曲线后 limits 未收缩"
        assert lo == pytest.approx(0.0, abs=1e-9)
    def test_revive_with_transform_view_expansion_uses_projected(self, plot_factory):
        """评审 P2-4：复活带换算时视图扩张须用投影 y_plot 而非 raw。"""
        pw = plot_factory()
        ci = _add_raw(pw)  # raw [1,2,3]
        ci.factor, ci.offset = 2.0, 10.0
        pw.demote_to_invalid("a", ci.color, ci.visible)
        assert pw.add_variable_to_plot(
            "a", x_values=np.arange(3, dtype=float),
            y_values=np.array([1.0, 2.0, 3.0]),
        )
        new_ci = pw.curves["a"]
        assert new_ci.factor == 2.0 and new_ci.offset == 10.0
        view_lo, view_hi = _view_y(pw)
        assert view_lo > 3.0, f"视图扩张误用原始值: {view_lo}"
        assert view_lo <= 12.0 and view_hi >= 16.0
