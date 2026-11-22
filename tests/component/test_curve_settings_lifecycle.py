"""显示名/换算设置的生命周期保真（编辑器强化轮 C1 / D3/D6）。

覆盖：构造点 y_raw 收口、demote 从旧 ci 捕获、占位复活带回、
_recreate_curve 重建不丢设置。reload saved_state 接力由 e2e 覆盖（C5/C6）。
"""

import numpy as np

from src.core.curve_transform import reproject


def _x():
    return np.arange(3, dtype=float)


def _y():
    return np.array([1.0, 2.0, 3.0])


def _add(pw, name="a"):
    assert pw.add_variable_to_plot(name, x_values=_x(), y_values=_y())
    return pw.curves[name]


class TestConstructionConvergence:
    def test_y_raw_holds_original(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        np.testing.assert_allclose(ci.y_raw, _y())
        np.testing.assert_allclose(ci.y_data, _y())
        assert ci.factor == 1.0 and ci.offset == 0.0 and ci.display_name == ""

    def test_transform_projects_curve_data(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.factor, ci.offset = 2.0, -10.0
        reproject(ci)
        ci.curve.setData(ci.x_data, ci.y_data)
        np.testing.assert_allclose(ci.curve.getData()[1], [-8.0, -6.0, -4.0])


class TestDemoteReviveCarry:
    def test_demote_captures_from_ci(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.display_name, ci.factor, ci.offset = "扭矩A", 2.0, -10.0
        pw.demote_to_invalid("a", ci.color, ci.visible)
        ref = pw.invalid_curves["a"]
        assert ref.display_name == "扭矩A"
        assert ref.factor == 2.0 and ref.offset == -10.0
        assert "a" not in pw.curves

    def test_demote_explicit_args_win(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.display_name = "旧名"
        pw.demote_to_invalid(
            "a", ci.color, ci.visible,
            display_name="新名", factor=3.0, offset=1.0,
        )
        ref = pw.invalid_curves["a"]
        assert ref.display_name == "新名"
        assert ref.factor == 3.0 and ref.offset == 1.0

    def test_revive_restores_settings_and_projection(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.display_name, ci.factor, ci.offset = "扭矩A", 2.0, -10.0
        pw.demote_to_invalid("a", ci.color, ci.visible)
        ci2 = _add(pw)  # 复活：消费占位
        assert "a" not in pw.invalid_curves
        assert ci2.display_name == "扭矩A"
        assert ci2.factor == 2.0 and ci2.offset == -10.0
        np.testing.assert_allclose(ci2.y_raw, _y())
        np.testing.assert_allclose(ci2.y_data, [-8.0, -6.0, -4.0])
        np.testing.assert_allclose(ci2.curve.getData()[1], [-8.0, -6.0, -4.0])


class TestRecreateFidelity:
    def test_recreate_keeps_display_name_and_transform(self, plot_factory):
        pw = plot_factory()
        ci = _add(pw)
        ci.display_name, ci.factor, ci.offset = "扭矩A", 2.0, -10.0
        reproject(ci)
        pw._multi_curve_manager._recreate_curve("a")
        new_ci = pw.curves["a"]
        assert new_ci.display_name == "扭矩A"
        assert new_ci.factor == 2.0 and new_ci.offset == -10.0
        np.testing.assert_allclose(new_ci.y_data, [-8.0, -6.0, -4.0])
        np.testing.assert_allclose(new_ci.curve.getData()[1], [-8.0, -6.0, -4.0])

    def test_recreate_default_transform_unchanged(self, plot_factory):
        pw = plot_factory()
        _add(pw)
        pw._multi_curve_manager._recreate_curve("a")
        new_ci = pw.curves["a"]
        np.testing.assert_allclose(new_ci.y_data, _y())
        assert new_ci.display_name == ""